//! Microphone for whisper mode (`lipflow/mic.py`): on only while the key is held, each chunk
//! timestamped on the same wall clock as the camera frames. Main thread owns the engine; the tap
//! runs on an audio thread and only appends to the shared chunk list.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2_avf_audio::{AVAudioEngine, AVAudioPCMBuffer, AVAudioTime};

use crate::audio::Chunks;

pub struct Mic {
    engine: Option<Retained<AVAudioEngine>>,
    chunks: Arc<Mutex<Chunks>>,
}

impl Mic {
    pub fn new() -> Self {
        Self { engine: None, chunks: Arc::new(Mutex::new(Chunks { rate: 48_000.0, chunks: Vec::new() })) }
    }

    /// Start listening (clears earlier audio). Errors are logged; whisper mode then reads lips only.
    pub fn start(&mut self) {
        if let Ok(mut c) = self.chunks.lock() {
            c.chunks.clear();
        }
        if self.engine.is_some() {
            return;
        }
        // SAFETY: AVAudioEngine setup on the main thread; the tap block only touches the shared
        // Arc<Mutex<Chunks>> and the buffer AVFoundation passes for the duration of the call.
        let started = unsafe {
            let engine = AVAudioEngine::new();
            let input = engine.inputNode();
            let format = input.outputFormatForBus(0);
            let rate = format.sampleRate();
            if let Ok(mut c) = self.chunks.lock() {
                c.rate = rate;
            }
            let sink = self.chunks.clone();
            let block = RcBlock::new(move |buf: NonNull<AVAudioPCMBuffer>, _when: NonNull<AVAudioTime>| {
                let buf = buf.as_ref();
                let n = buf.frameLength() as usize;
                let data = buf.floatChannelData();
                if data.is_null() || n == 0 {
                    return;
                }
                // first channel only (mono)
                let ch0 = (*data).as_ptr();
                let samples = std::slice::from_raw_parts(ch0, n).to_vec();
                let t0 = super::camera::now() - n as f64 / rate;
                if let Ok(mut c) = sink.lock() {
                    c.chunks.push((t0, samples));
                }
            });
            input.installTapOnBus_bufferSize_format_block(0, 1024, Some(&format), RcBlock::as_ptr(&block));
            engine.prepare();
            match engine.startAndReturnError() {
                Ok(()) => {
                    self.engine = Some(engine);
                    true
                }
                Err(e) => {
                    eprintln!("[lipflow] microphone unavailable: {}", e.localizedDescription());
                    input.removeTapOnBus(0);
                    false
                }
            }
        };
        let _ = started;
    }

    /// Stop and hand over what was captured.
    pub fn stop(&mut self) -> Chunks {
        if let Some(engine) = self.engine.take() {
            // SAFETY: stopping the engine this object started, on the main thread.
            unsafe {
                engine.inputNode().removeTapOnBus(0);
                engine.stop();
            }
        }
        let mut c = self.chunks.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        Chunks { rate: c.rate, chunks: std::mem::take(&mut c.chunks) }
    }
}
