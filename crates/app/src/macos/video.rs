//! Decode a video file to BGRA frames with AVFoundation (no ffmpeg/OpenCV).

use std::path::Path;

use anyhow::{Context, Result, bail};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_av_foundation::{AVAssetReader, AVAssetReaderOutput, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVMediaTypeVideo, AVURLAsset};
use objc2_core_foundation::CFString;
use objc2_core_media::{CMTime, CMTimeRange};
use objc2_core_video::{
    CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};

/// One decoded frame: tightly packed BGRA.
pub struct VideoFrame {
    pub bgra: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// Seconds from the start of the file, as `start + i / fps` (like OpenCV's reader in Python).
    pub t: f64,
}

/// Frames between `start` and `end` seconds (`end` None = to the end of the file).
pub fn read_frames(path: &Path, start: f64, end: Option<f64>, mut on_frame: impl FnMut(VideoFrame) -> Result<()>) -> Result<f64> {
    let url = NSURL::from_file_path(path).context("bad video path")?;
    // SAFETY: plain AVFoundation calls on objects we own; deprecated synchronous track access
    // is fine for a command-line reader (it blocks until the asset's tracks are loaded).
    unsafe {
        let asset = AVURLAsset::URLAssetWithURL_options(&url, None);
        let media = AVMediaTypeVideo.context("AVMediaTypeVideo unavailable")?;
        #[allow(deprecated)]
        let tracks = asset.tracksWithMediaType(media);
        let track = tracks.firstObject().context("no video track")?;
        let fps = f64::from(track.nominalFrameRate());
        let fps = if fps > 0.0 { fps } else { 25.0 };
        let reader = AVAssetReader::assetReaderWithAsset_error(&asset).map_err(|e| anyhow::anyhow!("AVAssetReader: {e:?}"))?;
        // SAFETY: CFString and NSString are toll-free bridged; the static outlives the dictionary.
        let key: &NSString = &*std::ptr::from_ref::<CFString>(kCVPixelBufferPixelFormatTypeKey).cast::<NSString>();
        let fmt = NSNumber::numberWithUnsignedInt(kCVPixelFormatType_32BGRA);
        let fmt_obj: &AnyObject = &fmt;
        let settings = NSDictionary::from_slices(&[key], &[fmt_obj]);
        let output = AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(&track, Some(&settings));
        output.setAlwaysCopiesSampleData(false);
        let start_t = CMTime::with_seconds(start, 600);
        let duration = match end {
            Some(e) => CMTime::with_seconds((e - start).max(0.0) + 1.0 / fps, 600),
            None => CMTime::with_seconds(1e7, 600),
        };
        reader.setTimeRange(CMTimeRange::new(start_t, duration));
        let out: Retained<AVAssetReaderOutput> = Retained::into_super(output);
        reader.addOutput(&out);
        if !reader.startReading() {
            bail!("could not start reading {}", path.display());
        }
        let mut i = 0usize;
        while let Some(sample) = out.copyNextSampleBuffer() {
            let t = start + i as f64 / fps;
            if end.is_some_and(|e| t > e) {
                break;
            }
            let Some(px) = sample.image_buffer() else { continue };
            CVPixelBufferLockBaseAddress(&px, CVPixelBufferLockFlags::ReadOnly);
            let (w, h, stride) = (CVPixelBufferGetWidth(&px), CVPixelBufferGetHeight(&px), CVPixelBufferGetBytesPerRow(&px));
            let base = CVPixelBufferGetBaseAddress(&px).cast::<u8>().cast_const();
            let mut bgra = Vec::with_capacity(w * h * 4);
            for y in 0..h {
                // SAFETY: the buffer is locked, BGRA, `h` rows of `stride` bytes, each row holding
                // at least `w * 4` bytes of pixels.
                bgra.extend_from_slice(std::slice::from_raw_parts(base.add(y * stride), w * 4));
            }
            CVPixelBufferUnlockBaseAddress(&px, CVPixelBufferLockFlags::ReadOnly);
            on_frame(VideoFrame { bgra, width: w, height: h, t })?;
            i += 1;
        }
        if reader.status() == AVAssetReaderStatus::Failed {
            bail!("decoding failed: {:?}", reader.error());
        }
        Ok(fps)
    }
}
