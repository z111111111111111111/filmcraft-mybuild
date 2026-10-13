//! Object selection for masks: a classic magic wand.
//!
//! A seed (a click, or the centre of a dragged box) grows into the 8-connected region whose
//! colours stay within `tolerance` of the seed; the region's boundary is traced and simplified
//! into a polygon for an editable [`filmcraft_project::MaskPath`]. Feed it to a mask and track
//! it with `masks.track`. Pure CPU, no models; automatic person detection without a seed is a
//! subject-model follow-up, not this module.
//!
//! All arithmetic is bounds-checked: hostile dimensions or seeds are errors, never panics.

use filmcraft_project::MaskPath;
use filmcraft_geom::Vec2;

/// What went wrong, each with an actionable message for the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentError {
    /// The buffer size does not match `w * h * 4`, or a dimension is zero.
    InvalidInput,
    /// The seed lies outside the picture.
    SeedOutside,
    /// The grown region is empty or a speck (raise the tolerance, or seed inside the object).
    NothingFound,
    /// The boundary stays above the vertex cap even simplified (raise the tolerance to merge detail).
    TooComplex,
}

/// Largest picture dimension this module traces at full detail; finer work is wasted on a mask
/// polygon and only costs time.
pub const MAX_WORK_SIDE: usize = 1440;
/// Hard cap on polygon vertices (masks stay editable).
pub const MAX_VERTICES: usize = 512;
/// Regions below this many pixels are specks, not objects.
pub const MIN_AREA: usize = 16;

/// Grow the seed's region: the 8-connected set of pixels whose every channel stays within
/// `tolerance` of the seed (L∞ in RGB, alpha ignored). Returns the bitmask (1 = selected).
pub fn grow_region(rgba: &[u8], w: usize, h: usize, seed: (u32, u32), tolerance: u8) -> Result<Vec<u8>, SegmentError> {
    if w == 0 || h == 0 || rgba.len() != w.saturating_mul(h).saturating_mul(4) {
        return Err(SegmentError::InvalidInput);
    }
    let (sx, sy) = (seed.0 as usize, seed.1 as usize);
    if sx >= w || sy >= h {
        return Err(SegmentError::SeedOutside);
    }
    let at = |x: usize, y: usize| -> Option<[u8; 3]> {
        let i = y.checked_mul(w)?.checked_add(x)?.checked_mul(4)?;
        let px = rgba.get(i..i.checked_add(3)?)?;
        Some([px.get(0).copied().unwrap_or(0), px.get(1).copied().unwrap_or(0), px.get(2).copied().unwrap_or(0)])
    };
    let seed_px = at(sx, sy).ok_or(SegmentError::SeedOutside)?;
    let tol = tolerance as i16;
    let mut mask = vec![0u8; w.saturating_mul(h)];
    let mut stack = vec![sy.saturating_mul(w).saturating_add(sx)];
    let mut area = 0usize;
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        if mask.get(i).is_none_or(|m| *m != 0) {
            continue;
        }
        let Some(px) = at(x, y) else { continue };
        let close = (px[0] as i16 - seed_px[0] as i16).abs().max((px[1] as i16 - seed_px[1] as i16).abs()).max((px[2] as i16 - seed_px[2] as i16).abs()) <= tol;
        if !close {
            continue;
        }
        if let Some(m) = mask.get_mut(i) {
            *m = 1;
        }
        area = area.saturating_add(1);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx >= 0 && ny >= 0 {
                    if let (Some(nx), Some(ny)) = (usize::try_from(nx).ok(), usize::try_from(ny).ok()) {
                        if nx < w && ny < h {
                            stack.push(ny.saturating_mul(w).saturating_add(nx));
                        }
                    }
                }
            }
        }
    }
    if area < MIN_AREA {
        return Err(SegmentError::NothingFound);
    }
    Ok(mask)
}

/// Trace every boundary loop of the bitmask with marching squares on a zero-padded grid.
/// Vertices come out in doubled coordinates (half-pixel edges become integers), each loop as
/// one polygon. The longest loop is the outer boundary; shorter ones are holes.
fn trace_loops(mask: &[u8], w: usize, h: usize) -> Vec<Vec<(i32, i32)>> {
    // segments keyed by start point (doubled coords); each start has at most one segment.
    use std::collections::HashMap;
    let mut starts: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
    let filled = |x: i32, y: i32| -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let (x, y) = (x as usize, y as usize);
        if x >= w || y >= h {
            return false;
        }
        mask.get(y.saturating_mul(w).saturating_add(x)).is_some_and(|m| *m != 0)
    };
    // cell (x, y) spans corners (2x,2y)..(2x+2,2y+2) in doubled coords
    for y in 0..h.saturating_add(1) {
        for x in 0..w.saturating_add(1) {
            let (x, y) = (x as i32, y as i32);
            let (a, b, c, d) = (filled(x - 1, y - 1), filled(x, y - 1), filled(x, y), filled(x - 1, y));
            // corners: a = top-left, b = top-right, c = bottom-right, d = bottom-left.
            // edges: T = (2x,2y)-(2x+2,2y), R = (2x+2,2y)-(2x+2,2y+2),
            //        B = (2x,2y+2)-(2x+2,2y+2), L = (2x,2y)-(2x,2y+2).
            let (x2, y2) = (x * 2, y * 2);
            let mut seg = |x0: i32, y0: i32, x1: i32, y1: i32| {
                starts.insert((x0, y0), (x1, y1));
            };
            match (a, b, c, d) {
                (false, false, false, false) | (true, true, true, true) => {}
                // single corners: cut across the corner, keeping the filled side left (clockwise)
                (true, false, false, false) => seg(x2, y2 + 1, x2 + 1, y2),
                (false, true, false, false) => seg(x2 + 1, y2, x2 + 2, y2 + 1),
                (false, false, true, false) => seg(x2 + 2, y2 + 1, x2 + 1, y2 + 2),
                (false, false, false, true) => seg(x2 + 1, y2 + 2, x2, y2 + 1),
                (false, true, true, true) => seg(x2 + 1, y2, x2, y2 + 1),
                (true, false, true, true) => seg(x2 + 2, y2 + 1, x2 + 1, y2),
                (true, true, false, true) => seg(x2 + 1, y2 + 2, x2 + 2, y2 + 1),
                (true, true, true, false) => seg(x2, y2 + 1, x2 + 1, y2 + 2),
                // straight edges
                (true, true, false, false) => seg(x2, y2 + 1, x2 + 2, y2 + 1),
                (false, false, true, true) => seg(x2 + 2, y2 + 1, x2, y2 + 1),
                (true, false, false, true) => seg(x2 + 1, y2, x2 + 1, y2 + 2),
                (false, true, true, false) => seg(x2 + 1, y2 + 2, x2 + 1, y2),
                // saddles: split by the diagonal that keeps filled corners connected
                (true, false, true, false) => {
                    seg(x2, y2 + 1, x2 + 1, y2);
                    seg(x2 + 2, y2 + 1, x2 + 1, y2 + 2);
                }
                (false, true, false, true) => {
                    seg(x2 + 1, y2, x2 + 2, y2 + 1);
                    seg(x2 + 1, y2 + 2, x2, y2 + 1);
                }
            }
        }
    }
    // join segments into loops (each start belongs to exactly one loop in a clean field)
    let mut loops = Vec::new();
    while let Some(start) = starts.keys().next().copied() {
        let mut path = vec![start];
        let mut cur = start;
        loop {
            let Some(next) = starts.remove(&cur) else { break };
            if next == start {
                break;
            }
            path.push(next);
            cur = next;
            if path.len() > starts.len().saturating_add(8).saturating_add(w).saturating_add(h).saturating_mul(4).max(1024) {
                break;
            }
        }
        if path.len() >= 4 {
            loops.push(path);
        }
    }
    loops
}

/// Douglas–Peucker simplification (iterative: no recursion depth to bound).
fn simplify(pts: &[(f64, f64)], eps: f64) -> Vec<(f64, f64)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    if let Some(first) = keep.first_mut() {
        *first = true;
    }
    if let Some(last) = keep.last_mut() {
        *last = true;
    }
    let mut stack = vec![(0usize, pts.len().saturating_sub(1))];
    while let Some((a, b)) = stack.pop() {
        if b <= a.saturating_add(1) {
            continue;
        }
        let (ax, ay) = pts.get(a).copied().unwrap_or((0.0f64, 0.0f64));
        let (bx, by) = pts.get(b).copied().unwrap_or((0.0f64, 0.0f64));
        let (dx, dy) = (bx - ax, by - ay);
        let len = dx.hypot(dy).max(1e-9f64);
        let mut far = 0.0f64;
        let mut idx = a;
        for i in a.saturating_add(1)..b {
            let Some((px, py)) = pts.get(i).copied() else { continue };
            let dist = ((py - ay) * dx - (px - ax) * dy).abs() / len;
            if dist > far {
                far = dist;
                idx = i;
            }
        }
        if far > eps {
            if let Some(k) = keep.get_mut(idx) {
                *k = true;
            }
            stack.push((a, idx));
            stack.push((idx, b));
        }
    }
    pts.iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| *p).collect()
}

/// Bounding box of the selected pixels (fallback when the traced boundary degenerates).
fn bbox(mask: &[u8], w: usize, h: usize) -> Option<(f64, f64, f64, f64)> {
    let (mut x0, mut y0, mut x1, mut y1): (f64, f64, f64, f64) = (w as f64, h as f64, -1.0, -1.0);
    for y in 0..h {
        for x in 0..w {
            if mask.get(y.saturating_mul(w).saturating_add(x)).is_some_and(|m| *m != 0) {
                x0 = x0.min(x as f64);
                y0 = y0.min(y as f64);
                x1 = x1.max(x as f64 + 1.0);
                y1 = y1.max(y as f64 + 1.0);
            }
        }
    }
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// The seed's object as a polygon in picture pixels: grow, trace the longest boundary loop,
/// simplify to mask-friendly vertices (corner points). Falls back to the region's bounding box
/// when the boundary degenerates; errors stay actionable.
pub fn object_polygon(rgba: &[u8], w: usize, h: usize, seed: (u32, u32), tolerance: u8) -> Result<Vec<(f64, f64)>, SegmentError> {
    let mask = grow_region(rgba, w, h, seed, tolerance)?;
    let mut loops = trace_loops(&mask, w, h);
    loops.sort_by_key(|l| std::cmp::Reverse(l.len()));
    if let Some(outer) = loops.first() {
        let pts: Vec<(f64, f64)> = outer.iter().map(|(x, y)| (*x as f64 / 2.0, *y as f64 / 2.0)).collect();
        for eps in [1.0, 2.0] {
            let simple = simplify(&pts, eps);
            if simple.len() >= 3 && simple.len() <= MAX_VERTICES {
                return Ok(simple);
            }
        }
        if pts.len() > MAX_VERTICES {
            return Err(SegmentError::TooComplex);
        }
    }
    // degenerate boundary (a line, a dot row): the region's box still selects the object
    if let Some((x0, y0, x1, y1)) = bbox(&mask, w, h) {
        return Ok(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
    }
    Err(SegmentError::NothingFound)
}

/// A polygon as an editable mask path (corner vertices; round or smooth them afterwards).
pub fn polygon_to_path(poly: &[(f64, f64)]) -> MaskPath {
    MaskPath::polygon(&poly.iter().map(|(x, y)| Vec2::new(*x, *y)).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, paint: impl Fn(usize, usize) -> [u8; 3]) -> Vec<u8> {
        let mut px = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let c = paint(x, y);
                if let Some(slot) = px.get_mut(y * w * 4 + x * 4..y * w * 4 + x * 4 + 4) {
                    slot[0] = c[0];
                    slot[1] = c[1];
                    slot[2] = c[2];
                    slot[3] = 255;
                }
            }
        }
        px
    }

    #[test]
    fn solid_rect_selects_its_bounds() {
        let px = frame(40, 30, |x, y| if (10..30).contains(&x) && (8..22).contains(&y) { [200, 40, 40] } else { [10, 10, 10] });
        let poly = object_polygon(&px, 40, 30, (20, 15), 0).unwrap();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for (x, y) in &poly {
            x0 = x0.min(*x);
            y0 = y0.min(*y);
            x1 = x1.max(*x);
            y1 = y1.max(*y);
        }
        assert!((x0 - 10.0).abs() <= 1.0 && (y0 - 8.0).abs() <= 1.0, "{poly:?}");
        assert!((x1 - 30.0).abs() <= 1.0 && (y1 - 22.0).abs() <= 1.0, "{poly:?}");
        assert!(poly.len() <= 12, "a rectangle simplifies small: {}", poly.len());
    }

    #[test]
    fn tolerance_merges_close_colours() {
        let px = frame(30, 10, |x, _| if x < 15 { [100, 100, 100] } else { [110, 110, 110] });
        // exact: only the left half
        let left = grow_region(&px, 30, 10, (5, 5), 0).unwrap();
        assert_eq!(left.iter().filter(|m| **m != 0).count(), 15 * 10);
        // loose: the whole strip
        let all = grow_region(&px, 30, 10, (5, 5), 20).unwrap();
        assert_eq!(all.iter().filter(|m| **m != 0).count(), 30 * 10);
    }

    #[test]
    fn the_seed_picks_its_own_region() {
        let px = frame(20, 20, |x, _| if x < 10 { [200, 0, 0] } else { [0, 0, 200] });
        let red = grow_region(&px, 20, 20, (2, 2), 0).unwrap();
        assert_eq!(red.iter().filter(|m| **m != 0).count(), 10 * 20);
        let blue = grow_region(&px, 20, 20, (15, 15), 0).unwrap();
        assert_eq!(blue.iter().filter(|m| **m != 0).count(), 10 * 20);
    }

    #[test]
    fn hostile_inputs_are_errors() {
        let px = frame(8, 8, |_, _| [9, 9, 9]);
        assert_eq!(object_polygon(&px, 8, 8, (99, 99), 10), Err(SegmentError::SeedOutside));
        assert_eq!(object_polygon(&px, 8, 8, (0, 0), 10).is_ok(), true, "a flat frame selects all of it");
        assert_eq!(grow_region(&px[..10], 8, 8, (0, 0), 10), Err(SegmentError::InvalidInput));
        assert_eq!(grow_region(&px, 0, 8, (0, 0), 10), Err(SegmentError::InvalidInput));
        // a single distinct pixel is a speck, not an object
        let mut dot = frame(12, 12, |_, _| [5, 5, 5]);
        if let Some(slot) = dot.get_mut(6 * 12 * 4 + 6 * 4..6 * 12 * 4 + 6 * 4 + 3) {
            slot.copy_from_slice(&[250, 250, 250]);
        }
        assert_eq!(grow_region(&dot, 12, 12, (6, 6), 0), Err(SegmentError::NothingFound));
    }
}
