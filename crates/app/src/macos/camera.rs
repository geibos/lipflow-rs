//! Webcam capture with live face tracking (`lipflow/camera.py`), on AVFoundation.
//!
//! While a recording is active every frame is tracked and kept (face crop + timestamp + anchors),
//! so when you stop talking the clip is already aligned and only needs the model. The camera is
//! opened lazily and closed after `IDLE_CLOSE` seconds without a recording, so the green light
//! isn't on all day.
//!
//! Threads: the `AVCaptureSession` lives on its own control thread (opening blocks for hundreds of
//! milliseconds); frames arrive on a serial dispatch queue and only touch `Shared`.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use block2::RcBlock;
use dispatch2::DispatchQueue;
use lipflow_face::{FaceLandmarker, FaceResult, Frame};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{AnyThread, DefinedClass, define_class, msg_send};
use objc2_av_foundation::{
    AVAuthorizationStatus, AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceInput, AVCaptureOutput, AVCaptureSession,
    AVCaptureSessionPreset1280x720, AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate, AVMediaTypeMuxed, AVMediaTypeVideo,
};
use objc2_core_foundation::CFString;
use objc2_core_media::CMSampleBuffer;
use objc2_core_video::{
    CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};

use crate::pipeline::Recording;

const IDLE_CLOSE: f64 = 45.0;
/// First frames are often black while exposure settles.
const WARMUP_FRAMES: usize = 3;

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

#[derive(Clone, Debug)]
pub struct CameraInfo {
    pub name: String,
    pub id: String,
    pub builtin: bool,
}

fn devices() -> Vec<Retained<AVCaptureDevice>> {
    let mut out = Vec::new();
    for media in [
        // SAFETY: reading immutable AVFoundation constants.
        unsafe { AVMediaTypeVideo },
        unsafe { AVMediaTypeMuxed },
    ]
    .into_iter()
    .flatten()
    {
        // SAFETY: class method returning an autoreleased array of devices.
        #[allow(deprecated)]
        let arr: Retained<NSArray<AVCaptureDevice>> = unsafe { AVCaptureDevice::devicesWithMediaType(media) };
        out.extend(arr.iter());
    }
    // OpenCV's order (by uniqueID), so camera ids saved by the Python app keep working.
    // SAFETY: plain property getter.
    out.sort_by_key(|d| unsafe { d.uniqueID() }.to_string());
    out
}

/// Cameras in a stable order, with names and ids.
pub fn list_cameras() -> Vec<CameraInfo> {
    devices()
        .iter()
        .map(|d| {
            // SAFETY: plain property getters.
            let (name, id, ty) = unsafe { (d.localizedName().to_string(), d.uniqueID().to_string(), d.deviceType().to_string()) };
            CameraInfo { name, id, builtin: ty.contains("BuiltIn") }
        })
        .collect()
}

/// "auto" → the Mac's own camera (never an iPhone); a name or id → that camera.
pub fn resolve_camera(pref: &str) -> Option<CameraInfo> {
    let cams = list_cameras();
    if pref != "auto" && !pref.is_empty() {
        let p = pref.to_lowercase();
        if let Some(c) = cams.iter().find(|c| c.id == pref || c.name.to_lowercase().contains(&p)) {
            return Some(c.clone());
        }
    }
    if let Some(c) = cams.iter().find(|c| c.builtin) {
        return Some(c.clone());
    }
    cams.iter().find(|c| !c.name.to_lowercase().contains("iphone")).or(cams.first()).cloned()
}

pub fn authorization() -> AVAuthorizationStatus {
    // SAFETY: class method with a valid media type constant.
    unsafe { AVMediaTypeVideo.map_or(AVAuthorizationStatus::NotDetermined, |m| AVCaptureDevice::authorizationStatusForMediaType(m)) }
}

/// Show the system prompt (main thread). `done(granted)` runs on an arbitrary queue.
pub fn request_access(done: impl Fn(bool) + 'static) {
    // SAFETY: reading an immutable AVFoundation constant.
    let Some(media) = (unsafe { AVMediaTypeVideo }) else { return };
    let block = RcBlock::new(move |granted: Bool| done(granted.as_bool()));
    // SAFETY: valid media type; the block is copied by AVFoundation.
    unsafe { AVCaptureDevice::requestAccessForMediaType_completionHandler(media, &block) };
}

/// What the UI gets for every frame (camera queue).
pub struct FrameEvent<'a> {
    pub frame: &'a Frame<'a>,
    pub face: Option<&'a FaceResult>,
    pub recording: bool,
    pub rec_duration: f64,
}

pub type OnFrame = Box<dyn Fn(&FrameEvent) + Send + Sync>;

struct Shared {
    rec: Mutex<Option<Recording>>,
    tracker: Mutex<FaceLandmarker>,
    track_always: AtomicBool,
    /// Record colour face patches (MultiVSR, Russian) instead of grayscale crops.
    faces: AtomicBool,
    ready: AtomicBool,
    warm: AtomicUsize,
    /// Diagnostics for "the camera isn't sending frames": when a frame last arrived and when
    /// the capture session was last (re)started, on the `now()` clock (0 = never).
    last_frame: Mutex<f64>,
    opened_at: Mutex<f64>,
    last_used: Mutex<f64>,
    error: Mutex<Option<String>>,
    /// Wall clock when the camera object was made; tracker timestamps count from here.
    t0: f64,
    last_ms: AtomicI64,
    on_frame: OnFrame,
    control: SyncSender<Cmd>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding one of these locks leaves plain data behind; keep going.
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

enum Cmd {
    Open(String),
    Close,
}

pub struct DelegateIvars {
    shared: Arc<Shared>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements; the class has no Drop impl.
    #[unsafe(super = NSObject)]
    #[ivars = DelegateIvars]
    struct FrameDelegate;

    // SAFETY: NSObjectProtocol has no safety requirements.
    unsafe impl NSObjectProtocol for FrameDelegate {}

    // SAFETY: the method signature matches the protocol.
    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for FrameDelegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn did_output(&self, _output: &AVCaptureOutput, sample: &CMSampleBuffer, _connection: &AVCaptureConnection) {
            on_sample(&self.ivars().shared, sample);
        }
    }
);

impl FrameDelegate {
    fn new(shared: Arc<Shared>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(DelegateIvars { shared });
        // SAFETY: NSObject's init has this signature.
        unsafe { msg_send![super(this), init] }
    }
}

fn on_sample(sh: &Shared, sample: &CMSampleBuffer) {
    // SAFETY: the sample buffer is valid for the duration of the delegate call.
    let Some(px) = (unsafe { sample.image_buffer() }) else { return };
    let t = now();
    // SAFETY: lock/unlock pair around reading the BGRA base address of a buffer we hold.
    unsafe { CVPixelBufferLockBaseAddress(&px, CVPixelBufferLockFlags::ReadOnly) };
    let (w, h, stride) = (CVPixelBufferGetWidth(&px), CVPixelBufferGetHeight(&px), CVPixelBufferGetBytesPerRow(&px));
    let base = CVPixelBufferGetBaseAddress(&px).cast::<u8>().cast_const();
    if !base.is_null() && w > 0 && h > 0 {
        // SAFETY: while locked, the buffer holds `h` rows of `stride` bytes (BGRA as requested).
        let data = unsafe { std::slice::from_raw_parts(base, stride * h) };
        let frame = Frame::bgra(data, w, h, stride);
        process(sh, &frame, t);
    }
    // SAFETY: matches the lock above.
    unsafe { CVPixelBufferUnlockBaseAddress(&px, CVPixelBufferLockFlags::ReadOnly) };
}

fn process(sh: &Shared, frame: &Frame, t: f64) {
    *lock(&sh.last_frame) = t;
    let warm = sh.warm.fetch_add(1, Ordering::Relaxed) + 1;
    if warm == WARMUP_FRAMES {
        sh.ready.store(true, Ordering::Release);
    }
    let recording = lock(&sh.rec).is_some();
    let face = if recording || sh.track_always.load(Ordering::Relaxed) {
        // MediaPipe-style tracking needs strictly increasing timestamps.
        let ms = (((t - sh.t0) * 1000.0) as i64).max(sh.last_ms.load(Ordering::Relaxed) + 1);
        sh.last_ms.store(ms, Ordering::Relaxed);
        match lock(&sh.tracker).detect(frame, ms) {
            Ok(f) => f,
            Err(e) => {
                *lock(&sh.error) = Some(format!("face tracking failed: {e}"));
                None
            }
        }
    } else {
        None
    };
    let mut duration = 0.0;
    {
        let mut rec = lock(&sh.rec);
        if let Some(r) = rec.as_mut()
            && warm >= WARMUP_FRAMES
        {
            r.push(t, frame, face.as_ref());
            duration = r.duration();
        }
    }
    (sh.on_frame)(&FrameEvent { frame, face: face.as_ref(), recording, rec_duration: duration });
    if !recording && !sh.track_always.load(Ordering::Relaxed) && t - *lock(&sh.last_used) > IDLE_CLOSE {
        let _ = sh.control.try_send(Cmd::Close);
    }
}

/// The capture session, owned by the control thread.
/// "file:PATH@START": a video file played back in real time stands in for the webcam
/// (testing, demos), like `--camera FILE` in the Python app.
fn file_source(pref: &str) -> Option<(std::path::PathBuf, f64)> {
    let rest = pref.strip_prefix("file:").map(str::to_string).or_else(|| std::path::Path::new(pref).is_file().then(|| pref.to_string()))?;
    let (path, start) = match rest.rsplit_once('@') {
        Some((p, s)) if s.parse::<f64>().is_ok() => (p.to_string(), s.parse().unwrap_or(0.0)),
        _ => (rest, 0.0),
    };
    Some((std::path::PathBuf::from(path), start))
}

fn play_file(shared: Arc<Shared>, path: std::path::PathBuf, start: f64, stop: Arc<AtomicBool>) {
    let t0 = now();
    let result = super::video::read_frames(&path, start, None, |f| {
        if stop.load(Ordering::Relaxed) {
            anyhow::bail!("stopped");
        }
        let due = t0 + (f.t - start);
        let wait = due - now();
        if wait > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(wait));
        }
        let frame = Frame::bgra(&f.bgra, f.width, f.height, f.width * 4);
        process(&shared, &frame, now());
        Ok(())
    });
    if let Err(e) = result
        && !stop.load(Ordering::Relaxed)
    {
        *lock(&shared.error) = Some(format!("video file: {e}"));
    }
}

fn control_loop(shared: Arc<Shared>, rx: Receiver<Cmd>) {
    let queue = DispatchQueue::new("lipflow.camera.frames", None);
    let mut session: Option<Retained<AVCaptureSession>> = None;
    let mut file_stop: Option<Arc<AtomicBool>> = None;
    let mut current = String::new();
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Close => {
                if let Some(s) = session.take() {
                    // SAFETY: stopping a session this thread created.
                    unsafe { s.stopRunning() };
                }
                if let Some(f) = file_stop.take() {
                    f.store(true, Ordering::Relaxed);
                }
                shared.ready.store(false, Ordering::Release);
                shared.warm.store(0, Ordering::Relaxed);
                current.clear();
            }
            Cmd::Open(pref) => {
                if (session.is_some() || file_stop.is_some()) && current == pref {
                    continue;
                }
                if let Some(s) = session.take() {
                    // SAFETY: as above.
                    unsafe { s.stopRunning() };
                }
                if let Some(f) = file_stop.take() {
                    f.store(true, Ordering::Relaxed);
                }
                shared.warm.store(0, Ordering::Relaxed);
                if let Some((path, start)) = file_source(&pref) {
                    let stop = Arc::new(AtomicBool::new(false));
                    let (sh, st) = (shared.clone(), stop.clone());
                    let _ = std::thread::Builder::new().name("lipflow-video".into()).spawn(move || play_file(sh, path, start, st));
                    file_stop = Some(stop);
                    current = pref;
                    continue;
                }
                *lock(&shared.opened_at) = now();
                match open_session(&shared, &queue, &pref) {
                    Ok(s) => {
                        session = Some(s);
                        current = pref;
                        *lock(&shared.error) = None;
                    }
                    Err(e) => {
                        eprintln!("[camera] {e}");
                        *lock(&shared.error) = Some(e.to_string());
                    }
                }
            }
        }
    }
}

fn open_session(shared: &Arc<Shared>, queue: &DispatchQueue, pref: &str) -> anyhow::Result<Retained<AVCaptureSession>> {
    let info = resolve_camera(pref).ok_or_else(|| anyhow::anyhow!("No camera found"))?;
    let device = devices()
        .into_iter()
        // SAFETY: property getter.
        .find(|d| unsafe { d.uniqueID() }.to_string() == info.id)
        .ok_or_else(|| anyhow::anyhow!("Camera {} disappeared", info.name))?;
    // SAFETY: standard AVFoundation session setup on objects this function owns.
    unsafe {
        let input = AVCaptureDeviceInput::deviceInputWithDevice_error(&device)
            .map_err(|e| anyhow::anyhow!("Could not open the camera: {}. Allow Lipflow in Settings → Privacy & Security → Camera", e.localizedDescription()))?;
        let session = AVCaptureSession::new();
        session.beginConfiguration();
        if session.canSetSessionPreset(AVCaptureSessionPreset1280x720) {
            session.setSessionPreset(AVCaptureSessionPreset1280x720);
        }
        if !session.canAddInput(&input) {
            anyhow::bail!("The camera is busy");
        }
        session.addInput(&input);
        let output = AVCaptureVideoDataOutput::new();
        // SAFETY: CFString and NSString are toll-free bridged.
        let key: &NSString = &*std::ptr::from_ref::<CFString>(kCVPixelBufferPixelFormatTypeKey).cast::<NSString>();
        let fmt = NSNumber::numberWithUnsignedInt(kCVPixelFormatType_32BGRA);
        let fmt_obj: &AnyObject = &fmt;
        output.setVideoSettings(Some(&NSDictionary::from_slices(&[key], &[fmt_obj])));
        output.setAlwaysDiscardsLateVideoFrames(true);
        let delegate = FrameDelegate::new(shared.clone());
        output.setSampleBufferDelegate_queue(Some(ProtocolObject::from_ref(&*delegate)), Some(queue));
        // The output retains its delegate weakly in some OS versions: keep ours alive with the session.
        std::mem::forget(delegate);
        session.addOutput(&output);
        session.commitConfiguration();
        session.startRunning();
        Ok(session)
    }
}

pub struct Camera {
    shared: Arc<Shared>,
    pub source: Mutex<String>,
}

impl Camera {
    pub fn new(source: &str, tracker: FaceLandmarker, on_frame: OnFrame) -> Arc<Self> {
        let (tx, rx) = sync_channel(8);
        let shared = Arc::new(Shared {
            rec: Mutex::new(None),
            tracker: Mutex::new(tracker),
            track_always: AtomicBool::new(false),
            faces: AtomicBool::new(false),
            ready: AtomicBool::new(false),
            warm: AtomicUsize::new(0),
            last_frame: Mutex::new(0.0),
            opened_at: Mutex::new(0.0),
            last_used: Mutex::new(now()),
            error: Mutex::new(None),
            t0: now(),
            last_ms: AtomicI64::new(-1),
            on_frame,
            control: tx,
        });
        let sh = shared.clone();
        std::thread::Builder::new()
            .name("lipflow-camera".into())
            .spawn(move || control_loop(sh, rx))
            .map(drop)
            .unwrap_or_else(|e| eprintln!("[camera] could not start the camera thread: {e}"));
        Arc::new(Self { shared, source: Mutex::new(source.to_string()) })
    }

    pub fn ensure_open(&self) {
        *lock(&self.shared.last_used) = now();
        let src = lock(&self.source).clone();
        let _ = self.shared.control.try_send(Cmd::Open(src));
    }

    pub fn set_source(&self, src: &str) {
        *lock(&self.source) = src.to_string();
        let _ = self.shared.control.try_send(Cmd::Close);
    }

    pub fn close(&self) {
        let _ = self.shared.control.try_send(Cmd::Close);
    }

    pub fn ready(&self) -> bool {
        self.shared.ready.load(Ordering::Acquire)
    }

    /// What the camera has been doing, for the log when frames don't arrive.
    pub fn diagnostics(&self) -> String {
        let t = now();
        let ago = |v: f64| if v > 0.0 { format!("{:.1}s ago", t - v) } else { "never".into() };
        format!(
            "last frame {}, session started {}, warm-up frames {}, recording {}",
            ago(*lock(&self.shared.last_frame)),
            ago(*lock(&self.shared.opened_at)),
            self.shared.warm.load(Ordering::Relaxed),
            self.is_recording()
        )
    }

    pub fn error(&self) -> Option<String> {
        lock(&self.shared.error).clone()
    }

    pub fn set_track_always(&self, on: bool) {
        self.shared.track_always.store(on, Ordering::Relaxed);
        if on {
            self.ensure_open();
        }
    }

    pub fn set_faces(&self, on: bool) {
        self.shared.faces.store(on, Ordering::Relaxed);
    }

    pub fn start_recording(&self) {
        self.ensure_open();
        let mut r = lock(&self.shared.rec);
        *r = Some(if self.shared.faces.load(Ordering::Relaxed) { Recording::with_faces() } else { Recording::new() });
        drop(r);
        lock(&self.shared.tracker).reset();
    }

    pub fn stop_recording(&self) -> Option<Recording> {
        *lock(&self.shared.last_used) = now();
        lock(&self.shared.rec).take()
    }

    pub fn is_recording(&self) -> bool {
        lock(&self.shared.rec).is_some()
    }

    /// Run `f` on the live recording (e.g. to snapshot it for a preview).
    pub fn with_recording<R>(&self, f: impl FnOnce(&Recording) -> R) -> Option<R> {
        lock(&self.shared.rec).as_ref().map(f)
    }
}
