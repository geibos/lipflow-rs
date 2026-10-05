//! Auto-AVSR visual speech recognition (Conformer encoder, Transformer decoder + CTC,
//! Transformer LM) and its joint CTC/attention beam search, on candle.

pub mod beam;
mod conv_train;
pub mod ctc;
pub mod decoder;
pub mod encoder;
pub mod multivsr;
#[cfg(feature = "metal")]
pub mod metal_ops;
pub mod nn;
pub mod reader;
pub mod spm;
pub mod train;

pub use reader::{LipReader, ReaderOptions};
