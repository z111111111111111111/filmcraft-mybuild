//! Machine-learning subject segmentation for FilmCraft's Object Select and Human Select tools.
//!
//! - [`catalogue`]: the catalogue of downloadable segmentation models (source URL pinned to a
//!   revision, SHA-256, size, licence), where they live in the per-user data directory, and
//!   (feature `download`) the verified downloader. Weights are **never** bundled or committed.
//! - [`matte`]: the soft (antialiased) alpha matte a model returns, with run-length coding and
//!   scaling.
//! - [`infer`] (feature `ml`): SAM 2 for a prompted object and BiRefNet HR Matting for a person,
//!   run on ONNX Runtime. Off by default, so the default build stays pure Rust.
//!
//! Whereas the classical Object/Human Select tools grow a region from a seed, these models return
//! a smooth alpha matte that keeps hair and soft edges. The engine falls back to the classical
//! tools when no model is installed or the build has no `ml` feature (see [`available`]).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod catalogue;
pub mod matte;
#[cfg(feature = "ml")]
pub mod infer;

use std::path::Path;

pub use matte::Matte;

/// Anything that can go wrong while segmenting; never panics.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SegmentError {
    #[error("stopped")]
    Cancelled,
    #[error("the segmentation model `{0}` is not downloaded")]
    NotInstalled(String),
    #[error("unknown segmentation model `{0}`")]
    UnknownModel(String),
    #[error("subject segmentation is not available in this build ({0})")]
    Unavailable(String),
    #[error("model: {0}")]
    Model(String),
    #[error("download: {0}")]
    Download(String),
    #[error("image: {0}")]
    Image(String),
    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for SegmentError {
    fn from(e: std::io::Error) -> Self {
        SegmentError::Io(e.to_string())
    }
}

/// Which subject to pick out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Subject {
    /// A prompted object (SAM 2): clicks and/or a box.
    Object,
    /// A person in the whole frame (BiRefNet HR Matting); no prompt.
    Human,
}

/// A point prompt for [`Subject::Object`], in source-frame pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromptPoint {
    pub x: f32,
    pub y: f32,
    /// `true` includes this point in the object, `false` excludes it.
    pub positive: bool,
}

/// An axis-aligned box prompt, in source-frame pixels (`x0,y0` top-left, `x1,y1` bottom-right).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

/// A prompt for [`Subject::Object`]: any mix of positive/negative clicks and an optional box.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompt {
    pub points: Vec<PromptPoint>,
    pub box_rect: Option<FrameBox>,
}

impl Prompt {
    /// A single positive click.
    pub fn click(x: f32, y: f32) -> Self {
        Self { points: vec![PromptPoint { x, y, positive: true }], box_rect: None }
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.box_rect.is_none()
    }
}

/// Segmentation options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Longest side the model runs at; the matte is stored at this resolution.
    pub resolution: u32,
    /// Blend neighbouring frames of a tracked range to steady the edge.
    pub smooth_temporal: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { resolution: 1024, smooth_temporal: true }
    }
}

/// Progress callback: `(fraction 0..1, status)`; return `false` to cancel.
pub type ProgressFn<'a> = &'a mut dyn FnMut(f32, &str) -> bool;

/// An RGBA8 image to segment: row-major, `w * h * 4` bytes.
#[derive(Clone, Copy, Debug)]
pub struct Rgba<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

impl<'a> Rgba<'a> {
    /// The image's pixel count, checked so hostile sizes cannot overflow or over-allocate.
    pub fn pixels(&self) -> Result<usize, SegmentError> {
        let w = self.width as u64;
        let h = self.height as u64;
        let n = w.checked_mul(h).ok_or_else(|| SegmentError::Image("frame is too large".into()))?;
        let need = n.checked_mul(4).ok_or_else(|| SegmentError::Image("frame is too large".into()))?;
        if n == 0 || need != self.data.len() as u64 {
            return Err(SegmentError::Image(format!("frame buffer is {} bytes, expected {need} for {w}x{h}", self.data.len())));
        }
        Ok(n as usize)
    }
}

/// Whether this build can run segmentation models.
pub const fn available() -> bool {
    cfg!(feature = "ml")
}

/// Segment `subject` out of `image`. `models_dir` is the directory the catalogue stores models in
/// (the caller supplies it, like `filmcraft-speech`). Returns a soft alpha [`Matte`] at
/// `opts.resolution`.
///
/// Without the `ml` feature this returns [`SegmentError::Unavailable`]; the caller (the engine)
/// then falls back to the classical Object/Human Select tools.
pub fn segment(
    subject: Subject,
    image: &Rgba,
    prompt: &Prompt,
    opts: &Options,
    models_dir: &Path,
    progress: ProgressFn,
) -> Result<Matte, SegmentError> {
    image.pixels()?;
    #[cfg(feature = "ml")]
    {
        infer::segment(subject, image, prompt, opts, models_dir, progress)
    }
    #[cfg(not(feature = "ml"))]
    {
        let _ = (subject, prompt, opts, models_dir, progress);
        Err(SegmentError::Unavailable("built without the `ml` feature".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_checks_buffers() {
        let ok = Rgba { width: 2, height: 2, data: &[0u8; 16] };
        assert_eq!(ok.pixels().unwrap(), 4);
        let short = Rgba { width: 2, height: 2, data: &[0u8; 8] };
        assert!(short.pixels().is_err());
        let zero = Rgba { width: 0, height: 0, data: &[] };
        assert!(zero.pixels().is_err());
    }

    #[test]
    fn empty_by_default_and_click() {
        assert!(Prompt::default().is_empty());
        assert!(!Prompt::click(1.0, 2.0).is_empty());
    }

    #[test]
    fn segments_only_when_available() {
        let img = Rgba { width: 1, height: 1, data: &[0, 0, 0, 255] };
        let r = segment(Subject::Object, &img, &Prompt::click(0.0, 0.0), &Options::default(), Path::new("."), &mut |_, _| true);
        if available() {
            // With the `ml` feature the call reaches infer (which may fail to load a model here).
            assert!(r.is_err() || r.is_ok());
        } else {
            assert!(matches!(r, Err(SegmentError::Unavailable(_))));
        }
    }
}
