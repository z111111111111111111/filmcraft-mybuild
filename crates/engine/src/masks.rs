//! Effect masks: `masks.*` commands (add / remove / edit / select / list / track /
//! selectObject / selectHuman).
//!
//! A mask lives on an effect instance of a video clip (`effect`: index or id; Opacity masks cut
//! out the clip itself). Commands address it with `clip` (default: the selected mask's clip, else
//! the first selected video clip), `effect` (default: the selected mask's effect, else `opacity`)
//! and `mask` (index, default: the selected mask). Path geometry is in clip pixels. Edits at the
//! playhead add/replace a keyframe when the parameter is animated, like every effect parameter;
//! keyframe commands (`effects.addKeyframe`, `effects.toggleAnimation`, …) take `"mask": n` to
//! address `path` / `feather` / `opacity` / `expansion`. Every edit is one undo step (`merge`: a
//! key folds a continuous drag into one step).

use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::{ClipId, EffectInstance, Mask, MaskMode, MaskPath, MaskRaster, MaskVertex, ParamValue, TrackKind, TrackMethod};
use filmcraft_time::Tick;
use filmcraft_segment_ml as seg;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use crate::commands::{CommandSpec, bad, bool_p, clip_p, f64_p, has_seq, str_p, time_p, u64_p};
use crate::{EngineError, Result, Session};

type Run = fn(&mut Session, &Value) -> Result<Value>;

/// The mask selected for on-monitor editing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskSel {
    pub clip: ClipId,
    pub effect: usize,
    pub mask: usize,
}

fn spec(id: &'static str, label: &'static str, params: &'static str, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: has_seq, run, journal: true }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "masks.add",
            "Create Mask",
            r#"{"clip":id?,"effect":index|id?="opacity","shape":"ellipse"|"polygon"|"rectangle"|"bezier","path":[[x,y]…]|{vertices}?,"center":[x,y]?,"size":[w,h]?,"cornerRadius":px?}"#,
            add,
        ),
        spec("masks.remove", "Delete Mask", r#"{"clip":id?,"effect":index|id?,"mask":n?}"#, remove),
        spec(
            "masks.selectObject",
            "Select Object",
            r#"{"clip":id?,"effect":index|id?="opacity","point":[x,y]? (clip pixels: grow from here),"box":[x,y,w,h]? (clip pixels: leave background margin around the object; the background is cut from its border),"tolerance":0..100?=25 (colour leeway),"feather":px?=0,"engine":"auto"|"ml"|"classic"?=auto (AI model when installed),"preview":bool?=false (outline only: no mask, no track),"track":"forward"|"backward"? (track the new mask right away)}"#,
            |s, p| select_subject(s, p, false),
        ),
        spec(
            "masks.selectHuman",
            "Select Human",
            r#"{"clip":id?,"effect":index|id?="opacity","box":[x,y,w,h]? (clip pixels: drag around the person),"point":[x,y]? (clip pixels; one of `box`/`point` is required),"tolerance":0..100?=25,"feather":px?=2,"engine":"auto"|"ml"|"classic"?=auto (AI model when installed),"preview":bool?=false,"track":"forward"|"backward"|"none"?="forward" (the mask follows the person)}"#,
            |s, p| select_subject(s, p, true),
        ),
        spec(
            "masks.set",
            "Change Mask",
            r#"{"clip":id?,"effect":index|id?,"mask":n?,"name":str?,"inverted":bool?,"mode":"none|add|subtract|intersect|lighten|darken|difference"?,"trackMethod":"position|positionRotation|positionScaleRotation"?,"feather":px?,"opacity":pct?,"expansion":px?,"path":path?,"time":ticks?,"merge":key?}"#,
            set,
        ),
        spec(
            "masks.moveVertex",
            "Move Mask Vertex",
            r#"{"clip":id?,"effect":index|id?,"mask":n?,"vertex":n,"handle":"point"|"in"|"out"?,"to":[x,y]?|"delta":[dx,dy]?,"breakHandles":bool?,"merge":key?}"#,
            move_vertex,
        ),
        spec("masks.translate", "Move Mask", r#"{"clip":id?,"effect":index|id?,"mask":n?,"delta":[dx,dy],"merge":key?}"#, translate),
        spec("masks.addVertex", "Add Mask Vertex", r#"{"clip":id?,"effect":index|id?,"mask":n?,"after":n,"at":[x,y]}"#, add_vertex),
        spec("masks.removeVertex", "Delete Mask Vertex", r#"{"clip":id?,"effect":index|id?,"mask":n?,"vertex":n}"#, remove_vertex),
        spec(
            "masks.track",
            "Track Selected Mask",
            r#"{"clip":id?,"effect":index|id?,"mask":n?,"direction":"forward"|"backward"?,"frames":n?,"method":"position|positionRotation|positionScaleRotation"?,"wait":bool?}"#,
            track,
        ),
        spec("masks.toggleVertexSmooth", "Convert Mask Vertex", r#"{"clip":id?,"effect":index|id?,"mask":n?,"vertex":n}"#, toggle_smooth),
        spec("masks.roundCorners", "Round Mask Corners", r#"{"clip":id?,"effect":index|id?,"mask":n?,"radius":px,"vertices":[n]?}"#, round_corners),
        CommandSpec {
            id: "masks.select",
            label: "Select Mask",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id,"effect":index|id,"mask":n}|{"none":true}"#,
            enabled: has_seq,
            run: select,
            journal: false,
        },
        CommandSpec {
            id: "masks.list",
            label: "List Masks",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id?,"time":ticks?}"#,
            enabled: has_seq,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "masks.models",
            label: "Segmentation Models",
            menu: &[],
            shortcut: None,
            params: r#"{}"#,
            enabled: crate::commands::always,
            run: list_models,
            journal: false,
        },
        CommandSpec {
            id: "masks.downloadModel",
            label: "Download Segmentation Model",
            menu: &[],
            shortcut: None,
            params: r#"{"subject":"object"|"human"|"sam2-small"|"birefnet-hr-matting"?="object","wait":bool?}"#,
            enabled: crate::commands::always,
            run: download_model,
            journal: false,
        },
    ]
}

/// A mask path from JSON: the serialized [`MaskPath`] object, an array of vertices
/// (`{"p":[x,y],"in":[x,y]?,"out":[x,y]?}`), or an array of `[x, y]` / `{"x","y"}` points.
pub fn path_from_json(v: &Value) -> Option<MaskPath> {
    // The vertex forms first: serde also accepts `{"p","in","out"}` vertices (the `masks.list` /
    // `masks.add` form) but silently drops the unknown `in` / `out` tangents.
    vertex_path(v).or_else(|| serde_json::from_value::<MaskPath>(v.clone()).ok())
}

fn vertex_path(v: &Value) -> Option<MaskPath> {
    let arr = v.as_array().or_else(|| v.get("vertices").and_then(Value::as_array))?;
    let closed = v.get("closed").and_then(Value::as_bool).unwrap_or(true);
    let mut vertices = Vec::new();
    for e in arr {
        if let Some(p) = e.get("p") {
            let t_in = e.get("in").or_else(|| e.get("t_in")).and_then(pt).unwrap_or_default();
            let t_out = e.get("out").or_else(|| e.get("t_out")).and_then(pt).unwrap_or_default();
            vertices.push(MaskVertex { p: pt(p)?, t_in, t_out });
        } else {
            vertices.push(MaskVertex::corner(pt(e)?));
        }
    }
    Some(MaskPath { vertices, closed })
}

fn pt(v: &Value) -> Option<Vec2> {
    if let Some(a) = v.as_array() {
        return Some(Vec2::new(a.first()?.as_f64()?, a.get(1)?.as_f64()?));
    }
    Some(Vec2::new(v.get("x")?.as_f64()?, v.get("y")?.as_f64()?))
}

pub fn path_to_json(p: &MaskPath) -> Value {
    json!({
        "closed": p.closed,
        "vertices": p.vertices.iter().map(|v| json!({"p": [v.p.x, v.p.y], "in": [v.t_in.x, v.t_in.y], "out": [v.t_out.x, v.t_out.y]})).collect::<Vec<_>>(),
    })
}

/// Resolve the target clip (and its source size in pixels).
fn target_clip(s: &Session, p: &Value) -> Result<ClipId> {
    if let Some(c) = clip_p(p, "clip") {
        return Ok(c);
    }
    if let Some(sel) = s.state.selected_mask {
        return Ok(sel.clip);
    }
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    s.state
        .selection
        .iter()
        .copied()
        .find(|c| seq.find_item(*c).and_then(|(t, _)| seq.track(t)).is_some_and(|t| t.kind == TrackKind::Video))
        .ok_or_else(|| bad("masks", "select a video clip (or pass `clip`)"))
}

fn effect_index(effects: &[EffectInstance], v: Option<&Value>, sel: Option<MaskSel>, clip: ClipId) -> Option<usize> {
    match v {
        Some(Value::Number(n)) => n.as_u64().map(|n| n as usize).filter(|i| *i < effects.len()),
        Some(Value::String(id)) => effects.iter().position(|e| &e.effect == id),
        _ => match sel {
            Some(s) if s.clip == clip && s.effect < effects.len() => Some(s.effect),
            _ => effects.iter().position(|e| e.effect == "opacity"),
        },
    }
}

/// Clip pixel size of a track item's source.
fn source_size(s: &Session, clip: ClipId) -> (f64, f64) {
    let seq = s.active_sequence();
    let it = seq.and_then(|q| q.find_item(clip)).map(|(_, i)| i);
    let dims = it.and_then(|i| s.project.item(i.item)).and_then(|pi| match &pi.kind {
        filmcraft_project::ItemKind::Media(m) => m.info.video.as_ref().map(|v| (v.width, v.height)),
        filmcraft_project::ItemKind::Sequence(q) => Some((q.settings.width, q.settings.height)),
        filmcraft_project::ItemKind::AdjustmentLayer { width, height, .. } | filmcraft_project::ItemKind::Graphic { width, height, .. } => {
            Some((*width, *height))
        }
        filmcraft_project::ItemKind::Subclip { parent, .. } => {
            s.project.item(*parent).and_then(|pp| pp.as_media()).and_then(|m| m.info.video.as_ref()).map(|v| (v.width, v.height))
        }
    });
    let (w, h) = dims.or_else(|| seq.map(|q| (q.settings.width, q.settings.height))).unwrap_or((1920, 1080));
    (w as f64, h as f64)
}

/// Run `f` on the addressed mask as one undoable edit. `f` gets the mask and the clip (media) time
/// of the edit.
fn edit_mask<R>(s: &mut Session, p: &Value, label: &str, f: impl FnOnce(&mut Mask, Tick) -> Result<R>) -> Result<R> {
    let clip = target_clip(s, p)?;
    let sel = s.state.selected_mask;
    let ph = time_p(s, p, "").unwrap_or(s.playhead());
    let eff = p.get("effect").cloned();
    let mask_i = u64_p(p, "mask").map(|m| m as usize).or(sel.filter(|x| x.clip == clip).map(|x| x.mask));
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let body = move |pr: &mut filmcraft_project::Project, _: &mut crate::EditorState| -> Result<R> {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let mt = it.source_time_at(ph.clamp(it.start, it.end() - Tick(1)));
        let ei = effect_index(&it.effects, eff.as_ref(), sel, clip).ok_or_else(|| bad(label, "no such effect on clip"))?;
        let e = &mut it.effects[ei];
        let mi = mask_i.ok_or_else(|| bad(label, "need `mask`"))?;
        let m = e.masks.get_mut(mi).ok_or_else(|| bad(label, format!("no mask {mi}")))?;
        f(m, mt)
    };
    match str_p(p, "merge") {
        Some(k) => {
            let key = format!("mask:{k}");
            s.edit_merged(label, &key, body)
        }
        None => s.edit(label, body),
    }
}

fn set_path_at(m: &mut Mask, t: Tick, path: MaskPath) {
    m.path.set_at(t, ParamValue::Path(path));
}

fn add(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let (w, h) = source_size(s, clip);
    let shape = str_p(p, "shape").unwrap_or("ellipse").to_ascii_lowercase();
    let center = p.get("center").and_then(pt).unwrap_or(Vec2::new(w / 2.0, h / 2.0));
    let r = w.min(h) * 0.2;
    let size = p.get("size").and_then(pt).unwrap_or(Vec2::new(r * 2.0, r * 2.0));
    let path = match p.get("path") {
        Some(v) => path_from_json(v).ok_or_else(|| bad("masks.add", "bad `path`"))?,
        None => match shape.as_str() {
            "ellipse" => MaskPath::ellipse(center, size * 0.5),
            "polygon" | "rectangle" | "rect" | "4-point" => {
                let (x0, y0, x1, y1) = (center.x - size.x / 2.0, center.y - size.y / 2.0, center.x + size.x / 2.0, center.y + size.y / 2.0);
                match f64_p(p, "cornerRadius") {
                    Some(r) if !r.is_finite() || r < 0.0 => return Err(bad("masks.add", "`cornerRadius` is pixels ≥ 0")),
                    Some(r) => MaskPath::rounded_rect(x0, y0, x1, y1, r),
                    None => MaskPath::rect(x0, y0, x1, y1),
                }
            }
            _ => return Err(bad("masks.add", "a bezier mask needs `path` (pen points)")),
        },
    };
    if path.len() < 3 && path.closed {
        return Err(bad("masks.add", "a mask needs at least 3 vertices"));
    }
    let eff = p.get("effect").cloned();
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let (ei, mi, name) = s.edit("Create Mask", move |pr, st| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let ei = effect_index(&it.effects, eff.as_ref(), None, clip).ok_or_else(|| bad("masks.add", "no such effect on clip"))?;
        let e = &mut it.effects[ei];
        if e.def().is_none_or(|d| d.kind != filmcraft_project::EffectKind::Video) {
            return Err(bad("masks.add", "masks apply to video effects"));
        }
        if e.effect == "motion" {
            return Err(bad("masks.add", "Motion has no masks"));
        }
        let n = e.masks.iter().filter_map(|m| m.name.strip_prefix("Mask (")?.strip_suffix(')')?.parse::<usize>().ok()).max().unwrap_or(0) + 1;
        let name = format!("Mask ({n})");
        e.masks.push(Mask::new(name.clone(), path));
        let mi = e.masks.len() - 1;
        st.selected_mask = Some(MaskSel { clip, effect: ei, mask: mi });
        st.selected_effect = None;
        Ok((ei, mi, name))
    })?;
    Ok(json!({"clip": clip.0, "effect": ei, "mask": mi, "name": name}))
}

/// Working width for object selection: full detail below it, scaled down above it (mask
/// polygons do not need 4K source pixels, and flood fill is linear in pixels).
const SELECT_MAX_WIDTH: f64 = 1440.0;

/// Push a mask like [`add`] does, with a caller-chosen name prefix ("Object", "Human").
/// Returns (effect index, mask index, name).
fn push_mask(s: &mut Session, clip: ClipId, eff: Option<Value>, prefix: &str, path: MaskPath, feather: f64, label: &str) -> Result<(usize, usize, String)> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    s.edit(label, move |pr, st| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let ei = effect_index(&it.effects, eff.as_ref(), None, clip).ok_or_else(|| bad(label, "no such effect on clip"))?;
        let e = &mut it.effects[ei];
        if e.def().is_none_or(|d| d.kind != filmcraft_project::EffectKind::Video) {
            return Err(bad(label, "masks apply to video effects"));
        }
        if e.effect == "motion" {
            return Err(bad(label, "Motion has no masks"));
        }
        let n = e
            .masks
            .iter()
            .filter_map(|m| m.name.strip_prefix(prefix).and_then(|r| r.strip_prefix(" (")?.strip_suffix(')')?.parse::<usize>().ok()).or_else(|| (m.name == prefix).then_some(0)))
            .max()
            .unwrap_or(0)
            + 1;
        let name = format!("{prefix} ({n})");
        let mut mask = Mask::new(name.clone(), path);
        if feather > 0.0 {
            mask.feather = filmcraft_project::Param::new(ParamValue::Float(feather));
        }
        e.masks.push(mask);
        let mi = e.masks.len() - 1;
        st.selected_mask = Some(MaskSel { clip, effect: ei, mask: mi });
        st.selected_effect = None;
        Ok((ei, mi, name))
    })
}

/// A selection seed in clip pixels: one point, or a box (x, y, w, h).
enum Seed {
    Point(Vec2),
    Box(f64, f64, f64, f64),
}

/// Which segmenter `masks.selectObject` / `masks.selectHuman` use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SegEngine {
    /// The AI model when it is installed, else the classical tools.
    Auto,
    /// Force the AI model (error when unavailable).
    Ml,
    /// Force the classical grow-from-seed tools.
    Classic,
}

/// The models directory when `subject`'s model is installed and this build can run it.
fn ml_model_dir(human: bool) -> Option<std::path::PathBuf> {
    if !seg::available() {
        return None;
    }
    let subject = if human { seg::Subject::Human } else { seg::Subject::Object };
    let dir = crate::transcript::models_dir()?;
    let m = seg::catalogue::for_subject(subject)?;
    seg::catalogue::installed(&dir, m).then_some(dir)
}

/// Whether the AI segmenter is ready for this subject (feature on and model installed).
fn ml_ready(human: bool) -> bool {
    ml_model_dir(human).is_some()
}

/// A human-readable reason for a segmentation failure.
fn ml_err(e: &seg::SegmentError) -> String {
    match e {
        seg::SegmentError::NotInstalled(_) => "the AI model is not downloaded (see masks.models / masks.downloadModel)".into(),
        seg::SegmentError::Cancelled => "stopped".into(),
        other => other.to_string(),
    }
}

/// The bounding box of the matte's covered pixels (alpha ≥ 128), or `None` when it is empty.
fn matte_bbox(m: &seg::Matte) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = (m.width(), m.height());
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if m.at(x, y).unwrap_or(0) >= 128 {
                any = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    any.then_some((x0, y0, x1, y1))
}

/// Segment with an AI model (SAM 2 for an object, BiRefNet HR Matting for a person) and store the
/// soft matte as a raster mask (schema v17). Runs on the CPU; the model is on disk already.
///
/// A `preview` shows the seed and defers running the model to the confirming call, so a click never
/// runs the model twice. AI masks are computed per frame, so no tracking is applied (the classic
/// tools track; a soft matte is re-segmented per frame instead).
fn select_ml(
    s: &mut Session,
    p: &Value,
    cmd: &str,
    human: bool,
    clip: ClipId,
    rgba: &[u8],
    fw: usize,
    fh: usize,
    sw: f64,
    sh: f64,
    seed: &Seed,
    track_dir: Option<String>,
) -> Result<Value> {
    let subject = if human { seg::Subject::Human } else { seg::Subject::Object };
    let dir = ml_model_dir(human).ok_or_else(|| bad(cmd, "the AI model is not downloaded (see masks.downloadModel)"))?;
    // clip pixels → frame pixels
    let (kx, ky) = (sw / fw as f64, sh / fh as f64);
    let to_frame = |x: f64, y: f64| (x / kx, y / ky);
    if p.get("preview").and_then(Value::as_bool).unwrap_or(false) {
        // show the seed as the outline without running the model
        let (x0, y0, x1, y1) = match seed {
            Seed::Point(q) => (q.x - 4.0, q.y - 4.0, q.x + 4.0, q.y + 4.0),
            Seed::Box(x, y, w, h) => (*x, *y, x + w, y + h),
        };
        return Ok(json!({
            "engine": "ml",
            "deferred": true,
            "outline": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "bbox": [x0, y0, x1, y1],
        }));
    }
    let prompt = match seed {
        Seed::Point(q) => {
            let (fx, fy) = to_frame(q.x, q.y);
            seg::Prompt::click(fx as f32, fy as f32)
        }
        Seed::Box(x, y, w, h) => {
            let (x0, y0) = to_frame(*x, *y);
            let (x1, y1) = to_frame(x + w, y + h);
            seg::Prompt { points: Vec::new(), box_rect: Some(seg::FrameBox { x0: x0 as f32, y0: y0 as f32, x1: x1 as f32, y1: y1 as f32 }) }
        }
    };
    let img = seg::Rgba { width: fw as u32, height: fh as u32, data: rgba };
    let opts = seg::Options { resolution: 1024, smooth_temporal: false };
    let matte = seg::segment(subject, &img, &prompt, &opts, &dir, &mut |_, _| true).map_err(|e| bad(cmd, ml_err(&e)))?;
    let (bx0, by0, bx1, by1) = matte_bbox(&matte).ok_or_else(|| bad(cmd, "the model found nothing to select: try a tighter box or a click inside the object"))?;
    let raster = MaskRaster { width: matte.width(), height: matte.height(), rle: matte.rle_encode() };
    // frame pixels → clip pixels for the reported bounds
    let (cx0, cy0) = (bx0 as f64 * kx, by0 as f64 * ky);
    let (cx1, cy1) = (bx1 as f64 * kx, by1 as f64 * ky);
    let eff = p.get("effect").cloned();
    let prefix = if human { "Human" } else { "Object" };
    let label = if human { "Select Human" } else { "Select Object" };
    let (ei, mi, name) = push_raster_mask(s, clip, eff, prefix, raster, label)?;
    Ok(json!({
        "clip": clip.0, "effect": ei, "mask": mi, "name": name,
        "engine": "ml", "vertices": 0,
        "bbox": [cx0, cy0, cx1, cy1],
        "outline": [[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]],
        "track": Value::Null,
        "note": track_dir.map(|_| "AI masks are computed per frame; tracking is not applied"),
    }))
}

/// Push a soft raster mask like [`push_mask`] does, with a caller-chosen name prefix.
fn push_raster_mask(s: &mut Session, clip: ClipId, eff: Option<Value>, prefix: &str, raster: MaskRaster, label: &str) -> Result<(usize, usize, String)> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    s.edit(label, move |pr, st| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let ei = effect_index(&it.effects, eff.as_ref(), None, clip).ok_or_else(|| bad(label, "no such effect on clip"))?;
        let e = &mut it.effects[ei];
        if e.def().is_none_or(|d| d.kind != filmcraft_project::EffectKind::Video) {
            return Err(bad(label, "masks apply to video effects"));
        }
        if e.effect == "motion" {
            return Err(bad(label, "Motion has no masks"));
        }
        let n = e
            .masks
            .iter()
            .filter_map(|m| m.name.strip_prefix(prefix).and_then(|r| r.strip_prefix(" (")?.strip_suffix(')')?.parse::<usize>().ok()).or_else(|| (m.name == prefix).then_some(0)))
            .max()
            .unwrap_or(0)
            + 1;
        let name = format!("{prefix} ({n})");
        e.masks.push(Mask::from_raster(name.clone(), raster));
        let mi = e.masks.len() - 1;
        st.selected_mask = Some(MaskSel { clip, effect: ei, mask: mi });
        st.selected_effect = None;
        Ok((ei, mi, name))
    })
}

/// Select an object (or a person) as an editable vector mask: grow the seed into its region,
/// trace and simplify the boundary, add the mask, and optionally track it right away.
///
/// `human` only changes the preset: soft feather and tracking forward by default, like isolating
/// a subject over a background. Automatic person detection without a seed is a subject-model
/// follow-up; this command segments from the given `point` or `box` today.
fn select_subject(s: &mut Session, p: &Value, human: bool) -> Result<Value> {
    let cmd = if human { "masks.selectHuman" } else { "masks.selectObject" };
    let clip = target_clip(s, p)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad(cmd, "no such clip"))?;
    let it = it.clone();
    let ph = time_p(s, p, "").unwrap_or(s.playhead());
    let mt = it.source_time_at(ph.clamp(it.start, it.end() - Tick(1)));
    let (sw, sh) = source_size(s, clip);
    if sw < 1.0 || sh < 1.0 {
        return Err(bad(cmd, "the clip has no picture"));
    }
    // seed in clip pixels: a point, or a box (the background is cut from its border)
    let seed_box = match (p.get("point").and_then(pt), p.get("box")) {
        (Some(q), None) => Seed::Point(q),
        (None, Some(b)) => {
            let (x, y, w, h) = (b.get(0).and_then(Value::as_f64), b.get(1).and_then(Value::as_f64), b.get(2).and_then(Value::as_f64), b.get(3).and_then(Value::as_f64));
            match (x, y, w, h) {
                (Some(x), Some(y), Some(w), Some(h)) if w > 0.0 && h > 0.0 => Seed::Box(x, y, w, h),
                _ => return Err(bad(cmd, "`box` is [x, y, w, h] in clip pixels with w, h > 0")),
            }
        }
        _ => return Err(bad(cmd, "give exactly one of `point` ([x, y]) or `box` ([x, y, w, h])")),
    };
    // the seed must touch the picture at all
    match &seed_box {
        Seed::Point(q) if !(0.0..sw).contains(&q.x) || !(0.0..sh).contains(&q.y) => return Err(bad(cmd, "the seed is outside the picture")),
        Seed::Box(x, y, _, _) if *x >= sw || *y >= sh || *x < 0.0 || *y < 0.0 => return Err(bad(cmd, "the seed is outside the picture")),
        _ => {}
    }
    let tolerance = match f64_p(p, "tolerance") {
        Some(t) if !t.is_finite() || t < 0.0 || t > 100.0 => return Err(bad(cmd, "`tolerance` is 0..100")),
        Some(t) => t,
        None => 25.0,
    };
    // validated before anything changes: a bad `track` must not leave a mask behind
    let track_dir: Option<String> = match str_p(p, "track") {
        Some(d) => match d.to_ascii_lowercase().as_str() {
            "forward" | "backward" => Some(d.to_ascii_lowercase()),
            "none" => None,
            other => return Err(bad(cmd, format!("`track`: forward, backward or none (got `{other}`)"))),
        },
        None => human.then_some("forward".to_string()),
    };
    let feather = match f64_p(p, "feather") {
        Some(f) if !f.is_finite() || f < 0.0 => return Err(bad(cmd, "`feather` is pixels ≥ 0")),
        Some(f) => f,
        None => {
            if human {
                2.0
            } else {
                0.0
            }
        },
    };
    // which engine: the AI model when installed (`auto`), or the classical grow-from-seed tools
    let engine = match str_p(p, "engine") {
        Some(e) => match e.to_ascii_lowercase().as_str() {
            "auto" => SegEngine::Auto,
            "ml" | "ai" => SegEngine::Ml,
            "classic" => SegEngine::Classic,
            other => return Err(bad(cmd, format!("`engine`: auto, ml or classic (got `{other}`)"))),
        },
        None => SegEngine::Auto,
    };
    if engine == SegEngine::Ml && !seg::available() {
        return Err(bad(cmd, "AI segmentation is not available in this build (built without the `segment` feature)"));
    }
    // decode at working size; flood fill is linear in pixels
    let scale = (SELECT_MAX_WIDTH / sw.max(sh)).min(1.0) as f32;
    let src = s.source(it.item).ok_or_else(|| bad(cmd, "the clip has no media to select from"))?;
    let fr = src.video_frame(filmcraft_media::FrameRequest { time: mt, scale }).ok().ok_or_else(|| bad(cmd, "can't decode the frame at the playhead"))?;
    let (fw, fh) = (fr.width as usize, fr.height as usize);
    if fw == 0 || fh == 0 {
        return Err(bad(cmd, "can't decode the frame at the playhead"));
    }
    let rgba = fr.to_rgba8();
    // AI path: when the model is installed, use it (`auto`) unless the caller asked for `classic`
    if engine != SegEngine::Classic && ml_ready(human) {
        return select_ml(s, p, cmd, human, clip, &rgba, fw, fh, sw, sh, &seed_box, track_dir);
    }
    // frame pixels → clip pixels
    let (kx, ky) = (sw / fw as f64, sh / fh as f64);
    let to_clip = |poly: Vec<(f64, f64)>| poly.into_iter().map(|(x, y)| (x * kx, y * ky)).collect::<Vec<_>>();
    let threshold = (tolerance * 1.6).round().clamp(0.0, 255.0) as u8;
    let segment_err = |e| {
        bad(
            cmd,
            match e {
                filmcraft_render::segment::SegmentError::SeedOutside => "the seed is outside the picture",
                filmcraft_render::segment::SegmentError::NothingFound => "nothing found there: click inside the object, drag a box around it, or raise `tolerance`",
                filmcraft_render::segment::SegmentError::TooComplex => "too much detail: raise `tolerance` to merge it, or drag a tighter box",
                filmcraft_render::segment::SegmentError::InvalidInput => "can't read the frame at the playhead",
            },
        )
    };
    let poly = match seed_box {
        // a box cuts the background from its border and keeps the biggest island inside: this
        // handles multicolour objects on complex scenes, where the background is what matches
        Seed::Box(x, y, w, h) => {
            let to_frame = |v: f64, size: f64, full: usize| ((v / size * full as f64).round() as i64).clamp(0, full as i64).max(0) as u32;
            let fb = (to_frame(x, sw, fw), to_frame(y, sh, fh), to_frame(x + w, sw, fw).saturating_sub(to_frame(x, sw, fw)), to_frame(y + h, sh, fh).saturating_sub(to_frame(y, sh, fh)));
            to_clip(filmcraft_render::segment::boxed_object_polygon(&rgba, fw, fh, fb, threshold).map_err(segment_err)?)
        }
        // a point grows from one colour: escalate the tolerance when the first try is a speck,
        // so clicks on textured objects still land
        Seed::Point(q) => {
            let seed_f = (((q.x / sw * fw as f64).round() as i64).clamp(0, fw as i64 - 1).max(0) as u32, ((q.y / sh * fh as f64).round() as i64).clamp(0, fh as i64 - 1).max(0) as u32);
            let mut threshold = threshold;
            let poly = loop {
                match filmcraft_render::segment::object_polygon(&rgba, fw, fh, seed_f, threshold) {
                    Ok(poly) => break poly,
                    Err(filmcraft_render::segment::SegmentError::NothingFound) if threshold < 255 => {
                        threshold = threshold.saturating_add((threshold / 2).max(8)).min(255);
                        continue;
                    }
                    Err(e) => return Err(segment_err(e)),
                }
            };
            to_clip(poly)
        }
    };
    let path = filmcraft_render::segment::polygon_to_path(&poly);
    let vertices = path.len();
    // preview outlines without changing anything: the monitor draws the blue frame first,
    // OK runs the same call again to create the mask
    if p.get("preview").and_then(Value::as_bool).unwrap_or(false) {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        let mut outline = Vec::new();
        for v in &path.vertices {
            x0 = x0.min(v.p.x);
            y0 = y0.min(v.p.y);
            x1 = x1.max(v.p.x);
            y1 = y1.max(v.p.y);
            outline.push([v.p.x, v.p.y]);
        }
        return Ok(json!({"vertices": vertices, "outline": outline, "bbox": [x0, y0, x1, y1]}));
    }
    let eff = p.get("effect").cloned();
    let prefix = if human { "Human" } else { "Object" };
    let label = if human { "Select Human" } else { "Select Object" };
    let (ei, mi, name) = push_mask(s, clip, eff, prefix, path, feather, label)?;
    // optionally track the new mask right away (the analyze-then-track flow)
    let tracking = match track_dir {
        None => Value::Null,
        Some(dir) => {
            let tp = serde_json::json!({"clip": clip.0, "effect": ei, "mask": mi, "direction": dir});
            match track(s, &tp) {
                Ok(v) => v,
                Err(e) => serde_json::json!({"error": e.to_string()}),
            }
        }
    };
    Ok(json!({"clip": clip.0, "effect": ei, "mask": mi, "name": name, "vertices": vertices, "track": tracking}))
}

fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let sel = s.state.selected_mask;
    let eff = p.get("effect").cloned();
    let mask_i = u64_p(p, "mask").map(|m| m as usize).or(sel.filter(|x| x.clip == clip).map(|x| x.mask)).ok_or_else(|| bad("masks.remove", "need `mask`"))?;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    s.edit("Delete Mask", move |pr, st| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let ei = effect_index(&it.effects, eff.as_ref(), sel, clip).ok_or_else(|| bad("masks.remove", "no such effect on clip"))?;
        let e = &mut it.effects[ei];
        if mask_i >= e.masks.len() {
            return Err(bad("masks.remove", format!("no mask {mask_i}")));
        }
        e.masks.remove(mask_i);
        if st.selected_mask.is_some_and(|x| x.clip == clip && x.effect == ei) {
            st.selected_mask = None;
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    let path = match p.get("path") {
        Some(v) => Some(path_from_json(v).ok_or_else(|| bad("masks.set", "bad `path`"))?),
        None => None,
    };
    let mode = match str_p(p, "mode") {
        Some(m) => Some(MaskMode::from_name(m).ok_or_else(|| bad("masks.set", format!("unknown mode `{m}`")))?),
        None => None,
    };
    let method = match str_p(p, "trackMethod") {
        Some(m) => Some(TrackMethod::from_name(m).ok_or_else(|| bad("masks.set", format!("unknown tracking method `{m}`")))?),
        None => None,
    };
    let name = str_p(p, "name").map(str::to_string);
    let inverted = bool_p(p, "inverted");
    let nums: Vec<(&str, f64)> = ["feather", "opacity", "expansion"].into_iter().filter_map(|k| f64_p(p, k).map(|v| (k, v))).collect();
    edit_mask(s, p, "Change Mask", move |m, t| {
        if let Some(n) = name {
            m.name = n;
        }
        if let Some(i) = inverted {
            m.inverted = i;
        }
        if let Some(md) = mode {
            m.mode = md;
        }
        if let Some(me) = method {
            m.track_method = me;
        }
        for (k, v) in nums {
            let v = match k {
                "feather" => v.max(0.0),
                "opacity" => v.clamp(0.0, 100.0),
                _ => v,
            };
            if let Some(prm) = m.param_mut(k) {
                prm.set_at(t, ParamValue::Float(v));
            }
        }
        if let Some(path) = path {
            set_path_at(m, t, path);
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

fn move_vertex(s: &mut Session, p: &Value) -> Result<Value> {
    let vi = u64_p(p, "vertex").ok_or_else(|| bad("masks.moveVertex", "need `vertex`"))? as usize;
    let handle = str_p(p, "handle").unwrap_or("point").to_string();
    let to = p.get("to").and_then(pt);
    let delta = p.get("delta").and_then(pt);
    let broken = bool_p(p, "breakHandles").unwrap_or(false);
    if to.is_none() && delta.is_none() {
        return Err(bad("masks.moveVertex", "need `to` or `delta`"));
    }
    edit_mask(s, p, "Move Mask Vertex", move |m, t| {
        let mut path = m.path_at(t);
        let v = path.vertices.get_mut(vi).ok_or_else(|| bad("masks.moveVertex", format!("no vertex {vi}")))?;
        match handle.as_str() {
            "in" | "out" => {
                let cur = if handle == "in" { v.p + v.t_in } else { v.p + v.t_out };
                let target = to.unwrap_or_else(|| cur + delta.unwrap_or_default());
                let tan = target - v.p;
                if handle == "in" {
                    v.t_in = tan;
                    if !broken {
                        v.t_out = tan * -1.0;
                    }
                } else {
                    v.t_out = tan;
                    if !broken {
                        v.t_in = tan * -1.0;
                    }
                }
            }
            _ => {
                v.p = to.unwrap_or_else(|| v.p + delta.unwrap_or_default());
            }
        }
        set_path_at(m, t, path);
        Ok(())
    })?;
    Ok(Value::Null)
}

fn translate(s: &mut Session, p: &Value) -> Result<Value> {
    let d = p.get("delta").and_then(pt).ok_or_else(|| bad("masks.translate", "need `delta`"))?;
    edit_mask(s, p, "Move Mask", move |m, t| {
        let path = m.path_at(t).transformed(&Affine::translate(d.x, d.y));
        set_path_at(m, t, path);
        Ok(())
    })?;
    Ok(Value::Null)
}

fn add_vertex(s: &mut Session, p: &Value) -> Result<Value> {
    let after = u64_p(p, "after").ok_or_else(|| bad("masks.addVertex", "need `after`"))? as usize;
    let at = p.get("at").and_then(pt).ok_or_else(|| bad("masks.addVertex", "need `at`"))?;
    edit_mask(s, p, "Add Mask Vertex", move |m, t| {
        let n = m.path_at(t).vertices.len();
        // clamp against the length before any insertion, so the index stays valid in the grown path
        let i = (after + 1).min(n);
        if m.path.is_animated() && m.path.keyframes.len() > 1 {
            if n == 0 {
                return Err(bad("masks.addVertex", "the animated mask path has no vertices to insert after"));
            }
            // keep every keyframe interpolable: insert the vertex into all of them
            for k in &mut m.path.keyframes {
                if let ParamValue::Path(path) = &mut k.value {
                    let len = path.vertices.len();
                    if len == 0 {
                        continue;
                    }
                    let at_i = i.min(len);
                    let a = path.vertices[after.min(len - 1)].p;
                    let b = path.vertices[at_i % len].p;
                    path.vertices.insert(at_i, MaskVertex::corner(a.lerp(b, 0.5)));
                }
            }
        }
        let mut path = m.path_at(t);
        if m.path.keyframes.len() > 1 {
            if let Some(v) = path.vertices.get_mut(i) {
                *v = MaskVertex::corner(at);
            }
        } else {
            path.vertices.insert(i.min(path.vertices.len()), MaskVertex::corner(at));
        }
        set_path_at(m, t, path);
        Ok(())
    })?;
    Ok(Value::Null)
}

fn remove_vertex(s: &mut Session, p: &Value) -> Result<Value> {
    let vi = u64_p(p, "vertex").ok_or_else(|| bad("masks.removeVertex", "need `vertex`"))? as usize;
    edit_mask(s, p, "Delete Mask Vertex", move |m, t| {
        if m.path_at(t).len() <= 3 {
            return Err(bad("masks.removeVertex", "a mask keeps at least 3 vertices"));
        }
        let drop = |path: &mut MaskPath| {
            if vi < path.vertices.len() {
                path.vertices.remove(vi);
            }
        };
        drop_all(m, t, drop);
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Apply a topology change to the path value and every path keyframe.
fn drop_all(m: &mut Mask, _t: Tick, f: impl Fn(&mut MaskPath)) {
    if let ParamValue::Path(p) = &mut m.path.value {
        f(p);
    }
    for k in &mut m.path.keyframes {
        if let ParamValue::Path(p) = &mut k.value {
            f(p);
        }
    }
}

fn toggle_smooth(s: &mut Session, p: &Value) -> Result<Value> {
    let vi = u64_p(p, "vertex").ok_or_else(|| bad("masks.toggleVertexSmooth", "need `vertex`"))? as usize;
    edit_mask(s, p, "Convert Mask Vertex", move |m, t| {
        let mut path = m.path_at(t);
        let n = path.vertices.len();
        if vi >= n {
            return Err(bad("masks.toggleVertexSmooth", format!("no vertex {vi}")));
        }
        let v = path.vertices[vi];
        if v.is_corner() {
            // smooth: tangent along the neighbours' direction, a third of the way to them
            let prev = path.vertices[(vi + n - 1) % n].p;
            let next = path.vertices[(vi + 1) % n].p;
            let dir = (next - prev) * (1.0 / 6.0);
            path.vertices[vi] = MaskVertex::smooth(v.p, dir);
        } else {
            path.vertices[vi] = MaskVertex::corner(v.p);
        }
        set_path_at(m, t, path);
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Round the mask's sharp corners (Premiere 26's rounded rectangle masks): every corner between
/// two straight edges (or only the listed `vertices`) becomes a circular fillet of `radius`
/// pixels, shrunk to fit half of each edge. The same corners are rounded in every Mask Path
/// keyframe so an animated path keeps interpolating. Returns how many corners were rounded.
fn round_corners(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "masks.roundCorners";
    let radius = f64_p(p, "radius").filter(|r| r.is_finite() && *r > 0.0).ok_or_else(|| bad(ID, "need `radius` (pixels, > 0)"))?;
    let only: Option<Vec<usize>> = match p.get("vertices") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let list = v.as_array().ok_or_else(|| bad(ID, "`vertices` is a list of vertex indices"))?;
            let mut out = list
                .iter()
                .map(|i| i.as_u64().and_then(|i| usize::try_from(i).ok()).ok_or_else(|| bad(ID, "`vertices` is a list of vertex indices")))
                .collect::<Result<Vec<_>>>()?;
            out.sort_unstable();
            out.dedup();
            Some(out)
        }
    };
    let rounded = edit_mask(s, p, "Round Mask Corners", move |m, t| {
        let mut corners = m.path_at(t).roundable_corners();
        if let Some(only) = &only {
            corners.retain(|i| only.binary_search(i).is_ok());
        }
        if corners.is_empty() {
            return Err(bad(ID, "no sharp corner between straight edges to round"));
        }
        drop_all(m, t, |path| *path = path.with_rounded_corners(radius, &corners));
        Ok(corners.len())
    })?;
    Ok(json!({"corners": rounded}))
}

fn select(s: &mut Session, p: &Value) -> Result<Value> {
    if bool_p(p, "none") == Some(true) || p.is_null() || p.as_object().is_some_and(|o| o.is_empty()) {
        s.state.selected_mask = None;
        return Ok(Value::Null);
    }
    let clip = target_clip(s, p)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad("masks.select", "no such clip"))?;
    let ei = effect_index(&it.effects, p.get("effect"), None, clip).ok_or_else(|| bad("masks.select", "no such effect on clip"))?;
    let mi = u64_p(p, "mask").unwrap_or(0) as usize;
    if mi >= it.effects[ei].masks.len() {
        return Err(bad("masks.select", format!("no mask {mi}")));
    }
    s.state.selected_mask = Some(MaskSel { clip, effect: ei, mask: mi });
    s.state.selected_effect = None;
    Ok(Value::Null)
}

/// Masks of a clip (or every clip with masks) evaluated at the playhead.
fn list(s: &mut Session, p: &Value) -> Result<Value> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let ph = time_p(s, p, "").unwrap_or(s.playhead());
    let only = clip_p(p, "clip");
    let mut out = Vec::new();
    for tr in &seq.video_tracks {
        for it in &tr.items {
            if only.is_some_and(|c| c != it.id) {
                continue;
            }
            let mt = it.source_time_at(ph.clamp(it.start, it.end() - Tick(1)));
            for (ei, e) in it.effects.iter().enumerate() {
                for (mi, m) in e.masks.iter().enumerate() {
                    out.push(json!({
                        "clip": it.id.0,
                        "effect": ei,
                        "effectId": e.effect,
                        "mask": mi,
                        "name": m.name,
                        "mode": m.mode.label(),
                        "inverted": m.inverted,
                        "trackMethod": m.track_method.label(),
                        "feather": m.feather.f64_at(mt),
                        "opacity": m.opacity.f64_at(mt),
                        "expansion": m.expansion.f64_at(mt),
                        "raster": m.raster.as_ref().map(|r| json!({"width": r.width, "height": r.height})),
                        "path": path_to_json(&m.path_at(mt)),
                        "pathKeyframes": m.path.keyframes.iter().map(|k| k.time.0).collect::<Vec<_>>(),
                        "selected": s.state.selected_mask == Some(MaskSel { clip: it.id, effect: ei, mask: mi }),
                    }));
                }
            }
        }
    }
    Ok(json!({"masks": out}))
}

/// The segmentation models: what they are, whether this build can run them, and if they are
/// installed (so a UI can offer to download the missing ones).
fn list_models(_s: &mut Session, _p: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    let models: Vec<Value> = seg::catalogue::catalogue()
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "name": m.name,
                "subject": m.subject,
                "description": m.description,
                "license": m.license,
                "licenseUrl": m.license_url,
                "author": m.author,
                "source": m.source,
                "size": m.size(),
                "installed": dir.as_deref().is_some_and(|d| seg::catalogue::installed(d, m)),
            })
        })
        .collect();
    Ok(json!({"available": seg::available(), "models": models}))
}

/// Download a segmentation model (object: SAM 2 small; human: BiRefNet HR Matting), SHA-256 and
/// size checked before it is used. Needs the `segment-download` feature.
fn download_model(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "masks.downloadModel";
    let key = str_p(p, "subject").unwrap_or("object").to_ascii_lowercase();
    let m = match key.as_str() {
        "object" | "sam2" => seg::catalogue::for_subject(seg::Subject::Object),
        "human" | "person" | "birefnet" => seg::catalogue::for_subject(seg::Subject::Human),
        id => seg::catalogue::find(id),
    }
    .ok_or_else(|| bad(cmd, format!("unknown model `{key}` (object, human, sam2-small or birefnet-hr-matting)")))?;
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for segmentation models"))?;
    #[cfg(not(feature = "segment-download"))]
    {
        let _ = (s, p, &dir, m);
        return Err(bad(cmd, "model downloads are not available in this build (built without the `segment-download` feature)"));
    }
    #[cfg(feature = "segment-download")]
    {
        if seg::catalogue::installed(&dir, m) {
            return Ok(json!({"model": m.id, "installed": true, "job": null}));
        }
        let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
        let job = new_job(s, format!("Downloading {}", m.name));
        job.progress.total.store(seg::catalogue::missing_bytes(&dir, m), std::sync::atomic::Ordering::Relaxed);
        let (prog, res, dirc, mc) = (job.progress.clone(), job.result.clone(), dir.clone(), m);
        let id = spawn(s, job, wait, "filmcraft-seg-download", move || {
            let t0 = web_time::Instant::now();
            let r = download_package_seg(&dirc, mc, &prog);
            finish_job(&prog, &res, r.map(|()| format!("{} installed", mc.name)), t0);
        })?;
        Ok(json!({"model": m.id, "installed": false, "job": id}))
    }
}

#[cfg(feature = "segment-download")]
fn download_package_seg(dir: &std::path::Path, m: &'static seg::catalogue::ModelInfo, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    seg::catalogue::download(dir, m, &mut progress).map_err(|e| e.to_string())
}

#[cfg(feature = "segment-download")]
fn new_job(s: &mut Session, label: String) -> crate::Job {
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    crate::Job { id, label, progress: Default::default(), result: Default::default() }
}

#[cfg(feature = "segment-download")]
fn spawn(s: &mut Session, job: crate::Job, wait: bool, name: &str, run: impl FnOnce() + Send + 'static) -> Result<u64> {
    let id = job.id;
    let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
    s.jobs.push(job);
    if wait || cfg!(target_arch = "wasm32") {
        run();
    } else {
        std::thread::Builder::new().name(name.into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    Ok(id)
}

#[cfg(feature = "segment-download")]
fn finish_job(prog: &filmcraft_export::Progress, res: &std::sync::Mutex<Option<std::result::Result<filmcraft_export::Report, String>>>, r: std::result::Result<String, String>, t0: web_time::Instant) {
    use std::sync::atomic::Ordering;
    use std::sync::PoisonError;
    let secs = t0.elapsed().as_secs_f64();
    let r = match r {
        Ok(status) => {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = status;
            Ok(filmcraft_export::Report { path: String::new(), frames: 0, seconds: secs, bytes: 0, render_fps: 0.0, extra_files: Vec::new() })
        }
        Err(e) => {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = e.clone();
            *prog.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
            Err(e)
        }
    };
    prog.finished.store(true, Ordering::Relaxed);
    *res.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
}

/// The parameter a keyframe / setParam command addresses: a mask parameter when `mask` is given.
pub(crate) fn target_param<'a>(e: &'a mut EffectInstance, p: &Value, pid: &str) -> Option<&'a mut filmcraft_project::Param> {
    match u64_p(p, "mask") {
        Some(mi) => e.masks.get_mut(mi as usize)?.param_mut(pid),
        None => e.params.get_mut(pid),
    }
}

// ---------------------------------------------------------------- tracking

/// A running mask-tracking job: tracked paths are written as Mask Path keyframes as they arrive
/// (one undo step for the whole track), see [`poll`].
pub struct PendingTrack {
    pub job: u64,
    pub target: MaskSel,
    /// Sequence the clip lives in.
    pub seq: filmcraft_project::ItemId,
    /// (media time, path) in tracking order.
    pub keys: Arc<Mutex<Vec<(Tick, MaskPath)>>>,
    applied: usize,
}

/// Longest side of the frames the tracker works on (speed; the result is in clip pixels).
const TRACK_MAX_WIDTH: f64 = 960.0;

fn track(s: &mut Session, p: &Value) -> Result<Value> {
    let clip = target_clip(s, p)?;
    let sel = s.state.selected_mask;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad("masks.track", "no such clip"))?;
    let it = it.clone();
    let ei = effect_index(&it.effects, p.get("effect"), sel, clip).ok_or_else(|| bad("masks.track", "no such effect on clip"))?;
    let mi = u64_p(p, "mask")
        .map(|m| m as usize)
        .or(sel.filter(|x| x.clip == clip && x.effect == ei).map(|x| x.mask))
        .ok_or_else(|| bad("masks.track", "need `mask`"))?;
    let mask = it.effects[ei].masks.get(mi).cloned().ok_or_else(|| bad("masks.track", format!("no mask {mi}")))?;
    if mask.is_raster() {
        return Err(bad("masks.track", "this is an AI (soft) mask, computed per frame: it cannot be tracked; run Object/Human Select on the frame instead"));
    }
    if s.mask_jobs.iter().any(|j| j.target.clip == clip && j.target.effect == ei && j.target.mask == mi) {
        return Err(bad("masks.track", "this mask is already being tracked"));
    }
    let method = match str_p(p, "method") {
        Some(m) => TrackMethod::from_name(m).ok_or_else(|| bad("masks.track", format!("unknown method `{m}`")))?,
        None => mask.track_method,
    };
    let backward = match str_p(p, "direction").unwrap_or("forward") {
        "forward" => false,
        "backward" => true,
        d => return Err(bad("masks.track", format!("direction `{d}`: forward or backward"))),
    };
    let rate = s.project.item(it.item).map(|i| i.frame_rate()).unwrap_or(seq.settings.frame_rate);
    let fd = rate.frame_duration();
    let ph = time_p(s, p, "").unwrap_or(s.playhead());
    let mt0 = it.source_time_at(ph.clamp(it.start, it.end() - Tick(1)));
    let (a, b) = (it.source_time_at(it.start), it.source_time_at(it.end() - Tick(1)));
    let (lo, hi) = (a.min(b), a.max(b));
    let room = if backward { (mt0 - lo).0 / fd.0.max(1) } else { (hi - mt0).0 / fd.0.max(1) };
    let steps = match u64_p(p, "frames") {
        Some(n) => (n as i64).min(room),
        None => room,
    }
    .max(0) as usize;
    if steps == 0 {
        return Err(bad("masks.track", if backward { "already at the clip's first frame" } else { "already at the clip's last frame" }));
    }
    let src = s.source(it.item).ok_or_else(|| EngineError::Other("the clip has no media to track".into()))?;
    let size = filmcraft_render::source_size(&s.project, it.item).ok_or_else(|| bad("masks.track", "the clip has no picture"))?;
    let scale = (TRACK_MAX_WIDTH / size.0.max(1) as f64).min(1.0) as f32;
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let label = format!("Track {} ({})", mask.name, if backward { "backward" } else { "forward" });
    let job = crate::Job { id, label: label.clone(), progress: Default::default(), result: Default::default() };
    job.progress.total.store(steps as u64, std::sync::atomic::Ordering::Relaxed);
    let keys: Arc<Mutex<Vec<(Tick, MaskPath)>>> = Arc::default();
    let (prog, res, out) = (job.progress.clone(), job.result.clone(), keys.clone());
    let start_path = mask.path_at(mt0);
    let run = move || {
        use std::sync::atomic::Ordering;
        let t0 = web_time::Instant::now();
        let gray = |t: Tick| -> Option<(filmcraft_render::track::Prepared, f64)> {
            let f = src.video_frame(filmcraft_media::FrameRequest { time: t, scale }).ok()?;
            let g = filmcraft_render::track::Gray::from_rgba8(f.width as usize, f.height as usize, &f.to_rgba8());
            Some((filmcraft_render::track::Prepared::new(g), f.width as f64 / size.0.max(1) as f64))
        };
        let mut path = start_path;
        let mut err: Option<String> = None;
        let mut done = 0u64;
        match gray(mt0) {
            None => err = Some("can't decode the clip".into()),
            Some((mut prev, mut k_prev)) => {
                lock(&out).push((mt0, path.clone()));
                for step in 1..=steps {
                    if prog.cancel.load(Ordering::Relaxed) {
                        err = Some("stopped".into());
                        break;
                    }
                    let t = if backward { mt0 - Tick(fd.0 * step as i64) } else { mt0 + Tick(fd.0 * step as i64) };
                    let Some((next, k)) = gray(t) else {
                        err = Some(format!("can't decode frame {step}"));
                        break;
                    };
                    let region: Vec<Vec2> = path.flatten(0.5).into_iter().map(|q| q * k_prev).collect();
                    let Some(st) = filmcraft_render::track::track_step(&prev, &next, &region, method) else {
                        err = Some(format!("lost track after {} frame(s): not enough detail inside the mask", step - 1));
                        break;
                    };
                    // frame pixels → clip pixels
                    let m = Affine::scale(1.0 / k, 1.0 / k).then_apply(&st.transform).then_apply(&Affine::scale(k_prev, k_prev));
                    path = path.transformed(&m);
                    lock(&out).push((t, path.clone()));
                    done += 1;
                    prog.done.store(done, Ordering::Relaxed);
                    *lock(&prog.status) = format!("Frame {step} of {steps} · {} features", st.inliers);
                    prev = next;
                    k_prev = k;
                }
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let r = match &err {
            // stopping keeps what was tracked
            Some(e) if e != "stopped" => Err(e.clone()),
            _ => Ok(filmcraft_export::Report {
                path: String::new(),
                frames: done,
                seconds: secs,
                bytes: 0,
                render_fps: done as f64 / secs.max(1e-6),
                extra_files: Vec::new(),
            }),
        };
        *lock(&prog.status) = match &err {
            Some(e) => format!("{e} ({done} frame(s) tracked)"),
            None => format!("Tracked {done} frame(s) in {secs:.1}s"),
        };
        prog.finished.store(true, Ordering::Relaxed);
        *lock(&res) = Some(r);
    };
    let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
    s.jobs.push(job);
    s.mask_jobs.push(PendingTrack { job: id, target: MaskSel { clip, effect: ei, mask: mi }, seq: seq_id, keys, applied: 0 });
    let wait = crate::commands::bool_p(p, "wait").unwrap_or(false);
    if wait || cfg!(target_arch = "wasm32") {
        run();
        poll(s);
    } else {
        std::thread::Builder::new().name("filmcraft-mask-track".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    let st = s.jobs.iter().find(|j| j.id == id).map(crate::Job::to_json).unwrap_or(Value::Null);
    Ok(json!({"job": id, "frames": steps, "status": st}))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write newly tracked paths as Mask Path keyframes (one merged undo step per tracking job) and
/// drop finished jobs. Called from [`Session::poll_persistence`] (once per UI frame) and after
/// synchronous runs.
pub fn poll(s: &mut Session) {
    use std::sync::atomic::Ordering;
    let mut i = 0;
    while i < s.mask_jobs.len() {
        let finished = s.jobs.iter().find(|j| j.id == s.mask_jobs[i].job).is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let new: Vec<(Tick, MaskPath)> = {
            let pj = &s.mask_jobs[i];
            lock(&pj.keys)[pj.applied..].to_vec()
        };
        if !new.is_empty() {
            let n_new = new.len();
            let pj = &s.mask_jobs[i];
            let (target, seq_id, key) = (pj.target, pj.seq, format!("mask-track-{}", pj.job));
            let r = s.edit_merged("Track Mask", &key, move |pr, _| {
                let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
                let (_, it) = q.find_item_mut(target.clip).ok_or(filmcraft_edit::EditError::NoItem(target.clip))?;
                let m = it.effects.get_mut(target.effect).and_then(|e| e.masks.get_mut(target.mask)).ok_or_else(|| bad("masks.track", "the mask is gone"))?;
                for (t, path) in new {
                    m.path.put_keyframe(t, ParamValue::Path(path));
                }
                Ok(())
            });
            match r {
                Ok(()) => s.mask_jobs[i].applied += n_new,
                Err(e) => {
                    s.error_toast("masks.track", e.to_string());
                    if let Some(j) = s.jobs.iter().find(|j| j.id == s.mask_jobs[i].job) {
                        j.progress.cancel.store(true, Ordering::Relaxed);
                    }
                    s.mask_jobs.remove(i);
                    continue;
                }
            }
        }
        let total = lock(&s.mask_jobs[i].keys).len();
        if finished && s.mask_jobs[i].applied >= total {
            s.mask_jobs.remove(i);
            continue;
        }
        i += 1;
    }
}
