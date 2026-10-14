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
/// A region covering more than this share of the picture leaked into the background.
pub const MAX_COVERAGE: f64 = 0.95;

/// Grow the seed's region: the 8-connected set of pixels whose every channel stays within
/// `tolerance` of the seed (L∞ in RGB, alpha ignored). Returns the bitmask (1 = selected).
pub fn grow_region(rgba: &[u8], w: usize, h: usize, seed: (u32, u32), tolerance: u8) -> Result<Vec<u8>, SegmentError> {
    grow_multi(rgba, w, h, std::slice::from_ref(&seed), tolerance)
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

/// Keep only the largest 8-connected component of the bitmask: after a box-background cut
/// the foreground may be islands, and the object is the biggest one. Every pixel is visited
/// once, so this stays linear.
pub fn largest_component(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w.saturating_mul(h);
    if w == 0 || h == 0 || mask.len() != n {
        return Vec::new();
    }
    let mut label = vec![0u32; n];
    let mut stack = Vec::new();
    let (mut best_label, mut best_size) = (0u32, 0usize);
    let mut next = 1u32;
    for s in 0..n {
        if mask.get(s).is_none_or(|m| *m == 0) || label.get(s).is_some_and(|l| *l != 0) {
            continue;
        }
        stack.clear();
        stack.push(s);
        if let Some(slot) = label.get_mut(s) {
            *slot = next;
        }
        let mut size = 0usize;
        while let Some(i) = stack.pop() {
            size = size.saturating_add(1);
            let (x, y) = (i % w, i / w);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 {
                        continue;
                    }
                    let j = (ny as usize).saturating_mul(w).saturating_add(nx as usize);
                    if j >= n || mask.get(j).is_none_or(|m| *m == 0) || label.get(j).is_some_and(|l| *l != 0) {
                        continue;
                    }
                    if let Some(slot) = label.get_mut(j) {
                        *slot = next;
                    }
                    stack.push(j);
                }
            }
        }
        if size > best_size {
            best_size = size;
            best_label = next;
        }
        next = next.saturating_add(1);
    }
    if best_label == 0 {
        return vec![0u8; n];
    }
    label.iter().map(|l| u8::from(*l == best_label)).collect()
}

/// Grow from any of `seeds` (shared visited set): one call floods a whole seed list, e.g.
/// every border pixel of a box.
pub fn grow_multi(rgba: &[u8], w: usize, h: usize, seeds: &[(u32, u32)], tolerance: u8) -> Result<Vec<u8>, SegmentError> {
    if w == 0 || h == 0 || rgba.len() != w.saturating_mul(h).saturating_mul(4) {
        return Err(SegmentError::InvalidInput);
    }
    let at = |x: usize, y: usize| -> Option<[u8; 3]> {
        let i = y.checked_mul(w)?.checked_add(x)?.checked_mul(4)?;
        let px = rgba.get(i..i.checked_add(3)?)?;
        Some([px.get(0).copied().unwrap_or(0), px.get(1).copied().unwrap_or(0), px.get(2).copied().unwrap_or(0)])
    };
    // every seed must read a colour: seeds outside are skipped, none inside is an error
    let colours: Vec<[u8; 3]> = seeds.iter().filter_map(|(sx, sy)| at(*sx as usize, *sy as usize)).collect();
    if colours.is_empty() {
        return Err(SegmentError::SeedOutside);
    }
    let tol = tolerance as i16;
    let mut mask = vec![0u8; w.saturating_mul(h)];
    let mut stack = Vec::new();
    for (sx, sy) in seeds {
        let (sx, sy) = (*sx as usize, *sy as usize);
        if sx < w && sy < h {
            stack.push(sy.saturating_mul(w).saturating_add(sx));
        }
    }
    let mut area = 0usize;
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        if mask.get(i).is_none_or(|m| *m != 0) {
            continue;
        }
        let Some(px) = at(x, y) else { continue };
        let close = colours.iter().any(|s| {
            (px[0] as i16 - s[0] as i16).abs().max((px[1] as i16 - s[1] as i16).abs()).max((px[2] as i16 - s[2] as i16).abs()) <= tol
        });
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

/// A bitmask as a polygon: longest boundary loop simplified, bounding-box fallback, same as
/// [`object_polygon`]'s tail. Shared by the point and box pipelines.
pub fn polygon_from_mask(mask: &[u8], w: usize, h: usize) -> Result<Vec<(f64, f64)>, SegmentError> {
    let mut loops = trace_loops(mask, w, h);
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
    if let Some((x0, y0, x1, y1)) = bbox(mask, w, h) {
        return Ok(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
    }
    Err(SegmentError::NothingFound)
}

/// The object inside `clip_box` (x, y, w, h, picture pixels): flood the box border as
/// background, keep the interior that is NOT background, trace the largest island. Unlike a
/// centre seed this handles multicolour objects on complex scenes — the common person case —
/// because the background, not the object, must match one tolerance.
pub fn boxed_object_polygon(rgba: &[u8], w: usize, h: usize, clip_box: (u32, u32, u32, u32), tolerance: u8) -> Result<Vec<(f64, f64)>, SegmentError> {
    if w == 0 || h == 0 || rgba.len() != w.saturating_mul(h).saturating_mul(4) {
        return Err(SegmentError::InvalidInput);
    }
    let (bx, by, bw, bh) = (clip_box.0 as usize, clip_box.1 as usize, clip_box.2 as usize, clip_box.3 as usize);
    if bw == 0 || bh == 0 || bx >= w || by >= h {
        return Err(SegmentError::SeedOutside);
    }
    let (x1, y1) = (bx.saturating_add(bw).min(w), by.saturating_add(bh).min(h));
    if x1 <= bx + 1 || y1 <= by + 1 {
        return Err(SegmentError::NothingFound);
    }
    // border pixels seed the background flood — but only the majority colour: a tight box
    // touches the object, and those pixels must not seed. Colours are bucketed coarsely so
    // video noise still votes together.
    let bucket = |px: [u8; 3]| ((px[0] >> 3) as u32, (px[1] >> 3) as u32, (px[2] >> 3) as u32);
    let border = |x: usize, y: usize| -> Option<[u8; 3]> {
        let i = y.checked_mul(w)?.checked_add(x)?.checked_mul(4)?;
        let px = rgba.get(i..i.checked_add(3)?)?;
        Some([px.get(0).copied().unwrap_or(0), px.get(1).copied().unwrap_or(0), px.get(2).copied().unwrap_or(0)])
    };
    use std::collections::HashMap;
    let mut votes: HashMap<(u32, u32, u32), usize> = HashMap::new();
    let mut edge: Vec<((u32, u32), [u8; 3])> = Vec::new();
    for x in bx..x1 {
        for y in [by, y1 - 1] {
            if let Some(px) = border(x, y) {
                *votes.entry(bucket(px)).or_insert(0) += 1;
                edge.push(((x as u32, y as u32), px));
            }
        }
    }
    for y in by.saturating_add(1)..y1.saturating_sub(1) {
        for x in [bx, x1 - 1] {
            if let Some(px) = border(x, y) {
                *votes.entry(bucket(px)).or_insert(0) += 1;
                edge.push(((x as u32, y as u32), px));
            }
        }
    }
    let bg_bucket = votes.into_iter().max_by_key(|(_, n)| *n).map(|(b, _)| b);
    let Some(bg_bucket) = bg_bucket else { return Err(SegmentError::NothingFound) };
    let seeds: Vec<(u32, u32)> = edge.into_iter().filter(|(_, px)| bucket(*px) == bg_bucket).map(|(p, _)| p).collect();
    let bg = grow_multi(rgba, w, h, &seeds, tolerance)?;
    let mut fg = vec![0u8; w.saturating_mul(h)];
    for y in by..y1 {
        for x in bx..x1 {
            let i = y.saturating_mul(w).saturating_add(x);
            if bg.get(i).is_none_or(|b| *b == 0) {
                if let Some(slot) = fg.get_mut(i) {
                    *slot = 1;
                }
            }
        }
    }
    let fg = largest_component(&fg, w, h);
    if fg.iter().all(|m| *m == 0) {
        return Err(SegmentError::NothingFound);
    }
    polygon_from_mask(&fg, w, h)
}

/// The seed's object as a polygon in picture pixels: grow, trace the longest boundary loop,
/// simplify to mask-friendly vertices (corner points). Falls back to the region's bounding box
/// when the boundary degenerates; errors stay actionable.
pub fn object_polygon(rgba: &[u8], w: usize, h: usize, seed: (u32, u32), tolerance: u8) -> Result<Vec<(f64, f64)>, SegmentError> {
    let mask = grow_region(rgba, w, h, seed, tolerance)?;
    polygon_from_mask(&mask, w, h)
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

    #[test]
    fn box_cut_keeps_a_multicolour_object() {
        // red disc + green square on a blue background: no single flood covers both, but the
        // box border is all background, so the box cut keeps exactly the two shapes' island
        let px = frame(40, 40, |x, y| {
            let in_disc = (x as i32 - 13).pow(2) + (y as i32 - 20).pow(2) < 64;
            let in_sq = (24..32).contains(&x) && (14..26).contains(&y);
            if in_disc {
                [200, 40, 40]
            } else if in_sq {
                [40, 200, 40]
            } else {
                [40, 40, 200]
            }
        });
        let poly = boxed_object_polygon(&px, 40, 40, (4, 8, 32, 24), 10).unwrap();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for (x, y) in &poly {
            x0 = x0.min(*x);
            y0 = y0.min(*y);
            x1 = x1.max(*x);
            y1 = y1.max(*y);
        }
        // the disc (the bigger island) roughly fills x 6..21, y 13..28
        assert!(x0 <= 6.0 && y0 <= 13.0 && x1 >= 20.0 && y1 >= 27.0, "{poly:?}");
        assert!(x1 <= 22.0 && y1 <= 29.0, "the square island is dropped, the disc kept: {poly:?}");
        // degenerate boxes are errors, not panics
        assert_eq!(boxed_object_polygon(&px, 40, 40, (0, 0, 0, 10), 10), Err(SegmentError::SeedOutside));
        assert_eq!(boxed_object_polygon(&px, 40, 40, (99, 99, 5, 5), 10), Err(SegmentError::SeedOutside));
    }

    #[test]
    fn largest_component_keeps_the_biggest_island() {
        // two filled squares, 4×4 and 2×2
        let mut mask = vec![0u8; 20 * 20];
        for y in 2..6 {
            for x in 2..6 {
                mask[y * 20 + x] = 1;
            }
        }
        for y in 12..14 {
            for x in 12..14 {
                mask[y * 20 + x] = 1;
            }
        }
        let kept = largest_component(&mask, 20, 20);
        assert_eq!(kept.iter().filter(|m| **m != 0).count(), 16);
        assert_eq!(kept[3 * 20 + 3], 1);
        assert_eq!(kept[12 * 20 + 12], 0);
        assert!(largest_component(&[], 0, 0).is_empty());
    }
}
