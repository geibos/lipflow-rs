//! Face tracking for Lipflow without MediaPipe: a TFLite interpreter for MediaPipe's face
//! detector and face mesh models, the FaceLandmarker tracking logic, and the mouth-crop
//! alignment of Auto-AVSR.

pub mod align;
pub mod face_crop;
mod gemm;
pub mod geom;
pub mod landmarker;
pub mod tflite;

pub use align::{Anchors, GrayFrame, anchors, mouth_open, mouth_rois};
pub use geom::Frame;
pub use landmarker::{FaceLandmarker, FaceResult};
