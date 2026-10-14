//! The soft alpha matte a segmentation model returns, and its run-length coding.
//!
//! A matte is an 8-bit coverage image (0 = fully transparent, 255 = fully opaque) at the model's
//! working resolution (see [`crate::Options::resolution`]). It is the raw output the engine turns
//! into a mask; the fields are private so every matte is a valid, consistently sized image.

use crate::SegmentError;

/// Largest matte we will build (pixels): 2^26 = 67 108 864, about 64 MiB of alpha. A hostile
/// resolution cannot make us allocate more than this.
const MAX_PIXELS: u64 = 1 << 26;

/// A soft (antialiased) 8-bit alpha matte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matte {
    width: u32,
    height: u32,
    /// Row-major coverage, `width * height` bytes.
    alpha: Vec<u8>,
}

fn pixel_count(width: u32, height: u32) -> Result<usize, SegmentError> {
    if width == 0 || height == 0 {
        return Err(SegmentError::Image("matte has no pixels".into()));
    }
    let n = (width as u64).checked_mul(height as u64).ok_or_else(|| SegmentError::Image("matte is too large".into()))?;
    if n > MAX_PIXELS {
        return Err(SegmentError::Image(format!("matte is too large ({n} pixels)")));
    }
    Ok(n as usize)
}

impl Matte {
    /// Build a matte from row-major coverage of exactly `width * height` bytes.
    pub fn new(width: u32, height: u32, alpha: Vec<u8>) -> Result<Self, SegmentError> {
        let n = pixel_count(width, height)?;
        if alpha.len() != n {
            return Err(SegmentError::Image(format!("matte has {} bytes, expected {n} for {width}x{height}", alpha.len())));
        }
        Ok(Self { width, height, alpha })
    }

    /// A matte that is fully opaque (`255`) everywhere.
    pub fn opaque(width: u32, height: u32) -> Result<Self, SegmentError> {
        let n = pixel_count(width, height)?;
        Ok(Self { width, height, alpha: vec![255; n] })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn alpha(&self) -> &[u8] {
        &self.alpha
    }

    pub fn into_alpha(self) -> (u32, u32, Vec<u8>) {
        (self.width, self.height, self.alpha)
    }

    pub fn len(&self) -> usize {
        self.alpha.len()
    }

    pub fn is_empty(&self) -> bool {
        self.alpha.is_empty()
    }

    /// Coverage at `(x, y)`, or `None` if out of bounds.
    pub fn at(&self, x: u32, y: u32) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.alpha.get((y as usize) * (self.width as usize) + (x as usize)).copied()
    }

    /// Mean coverage in `0.0..=1.0`.
    pub fn coverage(&self) -> f64 {
        if self.alpha.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.alpha.iter().map(|&a| a as u64).sum();
        sum as f64 / (self.alpha.len() as f64 * 255.0)
    }

    /// Run-length encode: repeated triples `[value, run_lo, run_hi]` where the run is a little-endian
    /// `u16` (`1..=65535`). Runs longer than `u16::MAX` are split into several triples. Storage stays
    /// small for the smooth alphas models produce.
    pub fn rle_encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let n = self.alpha.len();
        let mut i = 0usize;
        while i < n {
            let v = self.alpha[i];
            let mut run = 1usize;
            while i + run < n && self.alpha[i + run] == v && run < u16::MAX as usize {
                run += 1;
            }
            out.push(v);
            out.extend_from_slice(&(run as u16).to_le_bytes());
            i += run;
        }
        out
    }

    /// Inverse of [`Matte::rle_encode`]; rejects any encoding that does not cover exactly
    /// `width * height` pixels.
    pub fn rle_decode(width: u32, height: u32, bytes: &[u8]) -> Result<Self, SegmentError> {
        let n = pixel_count(width, height)?;
        if bytes.len() % 3 != 0 {
            return Err(SegmentError::Image("matte run data is not a whole number of runs".into()));
        }
        let mut alpha = Vec::with_capacity(n);
        for run in bytes.chunks_exact(3) {
            let value = run[0];
            let len = u16::from_le_bytes([run[1], run[2]]) as usize;
            if len == 0 {
                return Err(SegmentError::Image("matte has a zero-length run".into()));
            }
            if alpha.len() + len > n {
                return Err(SegmentError::Image("matte runs cover more pixels than the frame".into()));
            }
            alpha.resize(alpha.len() + len, value);
        }
        if alpha.len() != n {
            return Err(SegmentError::Image(format!("matte runs cover {} of {n} pixels", alpha.len())));
        }
        Ok(Self { width, height, alpha })
    }

    /// Bilinear rescale to `width * height` (used to fit the model's matte to a layer or preview).
    pub fn scaled_to(&self, width: u32, height: u32) -> Result<Self, SegmentError> {
        let n = pixel_count(width, height)?;
        if width == self.width && height == self.height {
            return Ok(self.clone());
        }
        let (sw, sh) = (self.width, self.height);
        let sx = sw as f32 / width as f32;
        let sy = sh as f32 / height as f32;
        let sample = |x: u32, y: u32| self.alpha[(y as usize) * (sw as usize) + (x as usize)] as f32;
        let mut out = vec![0u8; n];
        for y in 0..height {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
            let y0 = fy.floor() as u32;
            let y1 = (y0 + 1).min(sh - 1);
            let wy = fy - y0 as f32;
            for x in 0..width {
                let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
                let x0 = fx.floor() as u32;
                let x1 = (x0 + 1).min(sw - 1);
                let wx = fx - x0 as f32;
                let top = sample(x0, y0) * (1.0 - wx) + sample(x1, y0) * wx;
                let bottom = sample(x0, y1) * (1.0 - wx) + sample(x1, y1) * wx;
                let v = top * (1.0 - wy) + bottom * wy;
                out[(y as usize) * (width as usize) + (x as usize)] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(Self { width, height, alpha: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> Matte {
        let alpha = (0..(w * h)).map(|i| (i % 256) as u8).collect();
        Matte::new(w, h, alpha).unwrap()
    }

    #[test]
    fn new_validates_size() {
        assert!(Matte::new(0, 4, vec![]).is_err());
        assert!(Matte::new(2, 2, vec![0, 0, 0]).is_err());
        assert!(Matte::new(2, 2, vec![0, 0, 0, 0]).is_ok());
        // Too many pixels for the cap.
        assert!(Matte::new(1 << 14, 1 << 14, vec![]).is_err());
    }

    #[test]
    fn rle_round_trips() {
        for (w, h) in [(1u32, 1u32), (17, 3), (64, 64)] {
            let m = ramp(w, h);
            let enc = m.rle_encode();
            let dec = Matte::rle_decode(w, h, &enc).unwrap();
            assert_eq!(m, dec);
        }
    }

    #[test]
    fn rle_splits_long_runs() {
        let m = Matte::new(70000, 1, vec![7u8; 70000]).unwrap();
        let enc = m.rle_encode();
        // 70000 = 65535 + 4465 -> two runs -> six bytes.
        assert_eq!(enc.len(), 6);
        assert_eq!(Matte::rle_decode(70000, 1, &enc).unwrap(), m);
    }

    #[test]
    fn rle_decode_rejects_bad_data() {
        assert!(Matte::rle_decode(2, 2, &[1, 2]).is_err());
        assert!(Matte::rle_decode(2, 2, &[1, 0, 0]).is_err());
        assert!(Matte::rle_decode(2, 2, &[1, 5, 0]).is_err());
        assert!(Matte::rle_decode(2, 2, &[1, 3, 0]).is_err());
        assert!(Matte::rle_decode(2, 2, &[1, 1, 0, 1, 3, 0]).is_ok());
    }

    #[test]
    fn scaling_keeps_constant_and_bounds() {
        let flat = Matte::new(4, 4, vec![128; 16]).unwrap();
        assert_eq!(flat.scaled_to(9, 7).unwrap().alpha(), vec![128u8; 63]);
        let r = ramp(8, 8).scaled_to(3, 5).unwrap();
        assert_eq!((r.width(), r.height()), (3, 5));
        assert!(r.alpha().iter().all(|&a| a <= 255));
        assert_eq!(ramp(6, 6).scaled_to(6, 6).unwrap(), ramp(6, 6));
    }

    #[test]
    fn accessors() {
        let m = ramp(3, 2);
        assert_eq!(m.len(), 6);
        assert!(!m.is_empty());
        assert_eq!(m.at(2, 1), Some(m.alpha()[5]));
        assert_eq!(m.at(3, 0), None);
        let opaque = Matte::opaque(2, 2).unwrap();
        assert!((opaque.coverage() - 1.0).abs() < 1e-9);
    }
}
