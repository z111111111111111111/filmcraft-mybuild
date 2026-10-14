//! SAM 2 (object) and BiRefNet HR Matting (human) inference over ONNX Runtime (feature `ml`).

use std::path::Path;

use crate::{Matte, Options, ProgressFn, Prompt, Rgba, SegmentError, Subject};

/// Segment `subject` out of `image`; see [`crate::segment`].
pub fn segment(
    subject: Subject,
    _image: &Rgba,
    _prompt: &Prompt,
    _opts: &Options,
    _models_dir: &Path,
    _progress: ProgressFn,
) -> Result<Matte, SegmentError> {
    Err(SegmentError::Unavailable(format!("the `ml` backend is not wired up for {subject:?} in this build")))
}
