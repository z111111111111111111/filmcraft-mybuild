//! SAM 2 (object) and BiRefNet HR Matting (human) inference over ONNX Runtime (feature `ml`).
//!
//! Both models run on the CPU. SAM 2 takes a [`Prompt`] (positive/negative clicks and/or a box)
//! and returns the best candidate mask; BiRefNet HR Matting segments the person in the whole frame.
//! We feed the model an RGB frame no larger than `Options::resolution` on its longest side, so the
//! returned matte is bounded and matches the model's working resolution. Weights are local files
//! our downloader already verified (see [`crate::catalogue`]); the ONNX Runtime library ships next
//! to the executable.

use std::path::Path;

use usls::{
    models::{BiRefNet, SamPrompt, SAM2},
    Config, Image, Model,
};

use crate::{catalogue, Matte, Options, ProgressFn, Prompt, Rgba, SegmentError, Subject};

/// Segment `subject` out of `image`; see [`crate::segment`].
pub fn segment(
    subject: Subject,
    image: &Rgba,
    prompt: &Prompt,
    opts: &Options,
    models_dir: &Path,
    progress: ProgressFn,
) -> Result<Matte, SegmentError> {
    let max_side = opts.resolution.clamp(128, 1536);
    let (rgb, out_w, out_h) = fit_rgba(image, max_side)?;

    let model = catalogue::for_subject(subject).ok_or_else(|| SegmentError::Unavailable(format!("no model serves {subject:?}")))?;
    if !catalogue::installed(models_dir, model) {
        return Err(SegmentError::NotInstalled(model.id.to_string()));
    }
    if !progress(0.05, "Loading model") {
        return Err(SegmentError::Cancelled);
    }
    let dir = catalogue::model_dir(models_dir, model);

    let piece = |i: usize| -> Result<std::path::PathBuf, SegmentError> {
        let f = model.files.get(i).ok_or_else(|| SegmentError::Model(format!("{} is missing a model file", model.id)))?;
        Ok(dir.join(f.name))
    };

    let ys = match subject {
        Subject::Object => {
            let enc = piece(0)?;
            let dec = piece(1)?;
            let config = Config::sam2_small()
                .with_encoder_file(enc.to_string_lossy().into_owned())
                .with_decoder_file(dec.to_string_lossy().into_owned())
                .commit()
                .map_err(|e| SegmentError::Model(e.to_string()))?;
            let mut model = SAM2::new(config).map_err(|e| SegmentError::Model(e.to_string()))?;
            let img = Image::from_u8s(&rgb, out_w, out_h).map_err(|e| SegmentError::Image(e.to_string()))?;
            let sp = build_prompt(prompt, image.width, image.height, out_w, out_h)?;
            let xs = [img];
            let prompts = [sp];
            model.run((&xs, &prompts)).map_err(|e| SegmentError::Model(e.to_string()))?
        }
        Subject::Human => {
            let file = piece(0)?;
            let config = Config::birefnet_hr_matting()
                .with_model_file(file.to_string_lossy().into_owned())
                .commit()
                .map_err(|e| SegmentError::Model(e.to_string()))?;
            let mut model = BiRefNet::new(config).map_err(|e| SegmentError::Model(e.to_string()))?;
            let img = Image::from_u8s(&rgb, out_w, out_h).map_err(|e| SegmentError::Image(e.to_string()))?;
            let xs = [img];
            model.run(&xs).map_err(|e| SegmentError::Model(e.to_string()))?
        }
    };
    if !progress(0.9, "Finalising") {
        return Err(SegmentError::Cancelled);
    }

    let y = ys.into_iter().next().ok_or_else(|| SegmentError::Model("the model returned no result".into()))?;
    let mask = pick_mask(&y.masks, subject).ok_or_else(|| SegmentError::Model("the model returned no mask".into()))?;
    let matte = Matte::new(mask.width(), mask.height(), mask.to_vec())?;
    let matte = matte.scaled_to(out_w, out_h)?;
    progress(1.0, "Done");
    Ok(matte)
}

/// For an object, the largest covered mask (the object); for a person, the single matte the model
/// returns.
fn pick_mask<'a>(masks: &'a [usls::Mask], subject: Subject) -> Option<&'a usls::Mask> {
    match subject {
        Subject::Human => masks.first(),
        Subject::Object => masks
            .iter()
            .map(|m| (m, m.to_vec().iter().filter(|&&v| v >= 128).count()))
            .max_by_key(|(_, area)| *area)
            .map(|(m, _)| m),
    }
}

/// Build a SAM prompt, mapping source-frame coordinates into the resized frame.
fn build_prompt(prompt: &Prompt, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Result<SamPrompt, SegmentError> {
    if prompt.is_empty() {
        return Err(SegmentError::Model("object selection needs a click or a box".into()));
    }
    let (sx, sy) = (dst_w as f32 / src_w.max(1) as f32, dst_h as f32 / src_h.max(1) as f32);
    let mut sp = SamPrompt::default();
    if let Some(b) = prompt.box_rect {
        sp = sp.with_xyxy(b.x0 * sx, b.y0 * sy, b.x1 * sx, b.y1 * sy);
    }
    for p in &prompt.points {
        let (x, y) = (p.x * sx, p.y * sy);
        sp = if p.positive { sp.with_positive_point(x, y) } else { sp.with_negative_point(x, y) };
    }
    Ok(sp)
}

/// Convert the RGBA frame to RGB, downscaling (nearest neighbour) so the longest side is at most
/// `max_side`. Returns `(rgb, width, height)`.
fn fit_rgba(rgba: &Rgba, max_side: u32) -> Result<(Vec<u8>, u32, u32), SegmentError> {
    let (w, h) = (rgba.width, rgba.height);
    rgba.pixels()?;
    let long = w.max(h);
    if long <= max_side {
        let rgb = rgba.data.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        return Ok((rgb, w, h));
    }
    let scale = max_side as f32 / long as f32;
    let out_w = ((w as f32 * scale).round() as u32).max(1);
    let out_h = ((h as f32 * scale).round() as u32).max(1);
    let mut rgb = Vec::with_capacity((out_w as usize) * (out_h as usize) * 3);
    for y in 0..out_h {
        let sy = (((y as f64 + 0.5) / out_h as f64) * h as f64) as u32;
        let sy = sy.min(h - 1);
        for x in 0..out_w {
            let sx = (((x as f64 + 0.5) / out_w as f64) * w as f64) as u32;
            let sx = sx.min(w - 1);
            let i = ((sy as usize) * (w as usize) + (sx as usize)) * 4;
            rgb.push(rgba.data[i]);
            rgb.push(rgba.data[i + 1]);
            rgb.push(rgba.data[i + 2]);
        }
    }
    Ok((rgb, out_w, out_h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PromptPoint;

    #[test]
    fn fit_rgba_passes_small_through_and_drops_alpha() {
        let data = [10u8, 20, 30, 255, 40, 50, 60, 0];
        let img = Rgba { width: 2, height: 1, data: &data };
        let (rgb, w, h) = fit_rgba(&img, 1024).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(rgb, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn fit_rgba_downscales_to_max_side() {
        let data = vec![128u8; 8 * 4 * 4];
        let img = Rgba { width: 8, height: 4, data: &data };
        let (rgb, w, h) = fit_rgba(&img, 4).unwrap();
        assert_eq!((w, h), (4, 2));
        assert_eq!(rgb.len(), 4 * 2 * 3);
    }

    #[test]
    fn build_prompt_maps_coordinates() {
        let p = Prompt { points: vec![PromptPoint { x: 50.0, y: 25.0, positive: true }], box_rect: None };
        let sp = build_prompt(&p, 100, 100, 200, 50).unwrap();
        assert_eq!(sp.coords[0][0], [100.0, 12.5]);
        assert_eq!(sp.labels[0][0], 1.0);
        assert!(build_prompt(&Prompt::default(), 10, 10, 10, 10).is_err());
    }

    #[test]
    fn picks_largest_object_mask() {
        let small = usls::Mask::new(&[0u8, 255, 0, 0], 2, 2).unwrap();
        let large = usls::Mask::new(&[255u8; 9], 3, 3).unwrap();
        let masks = [small, large];
        let picked = pick_mask(&masks, Subject::Object).unwrap();
        assert_eq!(picked.dimensions(), (3, 3));
        assert_eq!(pick_mask(&masks, Subject::Human).unwrap().dimensions(), (2, 2));
    }
}
