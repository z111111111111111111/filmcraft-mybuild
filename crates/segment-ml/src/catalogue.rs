//! The segmentation model catalogue and the model store.
//!
//! Models are downloaded on first use, after the user confirms (the dialog shows the size, the
//! source URL and the licence), into `<data dir>/models/<id>/`. They are never bundled with
//! FilmCraft or committed to the repository. The weights are the ONNX conversions published in
//! `github.com/jamjamjon/assets` (the same files the `usls` library documents); every file is
//! pinned to its release URL and checked against its SHA-256 and size before it is used.
//!
//! The SHA-256 values here were computed from the release assets themselves (GitHub reports no
//! digest for them).
//!
//! Licences: SAM 2 is Apache-2.0 (Meta), BiRefNet is MIT (ZhengPeng7); the ONNX conversions are
//! permissive too. Weights are downloaded, never shipped.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::Subject;
#[cfg(feature = "download")]
use crate::SegmentError;

/// One file of a model.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// A downloadable segmentation model.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    /// Which tool this model serves.
    pub subject: Subject,
    pub description: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub author: &'static str,
    /// Human-readable source page.
    pub source: &'static str,
    pub files: &'static [ModelFile],
}

impl ModelInfo {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

macro_rules! asset {
    ($repo:literal, $name:literal, $sha:literal, $size:expr) => {
        ModelFile {
            name: $name,
            url: concat!("https://github.com/jamjamjon/assets/releases/download/", $repo, "/", $name),
            sha256: $sha,
            size: $size,
        }
    };
}

static CATALOGUE: &[ModelInfo] = &[
    ModelInfo {
        id: "sam2-small",
        name: "SAM 2 (small)",
        subject: Subject::Object,
        description: "Click the object, or drag a box around it; SAM 2 cuts it out. Prompt as many times as you like.",
        license: "Apache-2.0",
        license_url: "https://github.com/facebookresearch/sam2/blob/main/LICENSE",
        author: "Meta (SAM 2); ONNX conversion by jamjamjon",
        source: "https://github.com/facebookresearch/sam2",
        files: &[
            asset!("sam", "sam2-hiera-small-encoder.onnx", "c257d95cb8f3fc313b471a8dbe7f44cc63052f0398a4970cf4220ca6da7ac0ea", 162_703_539),
            asset!("sam", "sam2-hiera-small-decoder.onnx", "7a2dda4ecf380bf0e01159281ee3616c3e527507863446752830a1b8604281c6", 20_627_227),
        ],
    },
    ModelInfo {
        id: "birefnet-hr-matting",
        name: "BiRefNet HR Matting",
        subject: Subject::Human,
        description: "Whole-frame person matting with detail-preserving high resolution (8-bit weights).",
        license: "MIT",
        license_url: "https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        author: "ZhengPeng7 (BiRefNet); ONNX conversion by jamjamjon",
        source: "https://github.com/ZhengPeng7/BiRefNet",
        files: &[asset!("birefnet", "HR-matting-epoch-135-q8.onnx", "ec24c91be5e3da2d55da3e7a3baf2be762343c079d5505326b58e9fa32ece3b4", 641_123_831)],
    },
];

pub fn catalogue() -> &'static [ModelInfo] {
    CATALOGUE
}

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    CATALOGUE.iter().find(|m| m.id == id)
}

/// The model that serves `subject` (the only one per subject, for now).
pub fn for_subject(subject: Subject) -> Option<&'static ModelInfo> {
    CATALOGUE.iter().find(|m| m.subject == subject)
}

/// Where model `m` lives under `models_dir`.
pub fn model_dir(models_dir: &Path, m: &ModelInfo) -> PathBuf {
    models_dir.join(m.id)
}

/// All files present with their catalogue sizes (contents were verified when downloaded).
pub fn installed(models_dir: &Path, m: &ModelInfo) -> bool {
    let d = model_dir(models_dir, m);
    m.files.iter().all(|f| std::fs::metadata(d.join(f.name)).is_ok_and(|md| md.len() == f.size))
}

/// Bytes still to download for `m` (files already present are skipped).
pub fn missing_bytes(models_dir: &Path, m: &ModelInfo) -> u64 {
    let d = model_dir(models_dir, m);
    m.files.iter().filter(|f| !std::fs::metadata(d.join(f.name)).is_ok_and(|md| md.len() == f.size)).map(|f| f.size).sum()
}

/// Download progress: `(bytes done, bytes total, file)`; return false to cancel.
#[cfg(feature = "download")]
pub type DownloadProgress<'a> = &'a mut dyn FnMut(u64, u64, &str) -> bool;

/// Download the missing files of `m` into `models_dir/<id>/`: each file streams into `<name>.part`,
/// its SHA-256 is checked, then it is renamed into place. A wrong checksum deletes the part file
/// and fails. Pure Rust TLS (rustls + RustCrypto) with the operating system's certificate verifier.
#[cfg(feature = "download")]
pub fn download(models_dir: &Path, m: &ModelInfo, progress: DownloadProgress) -> Result<(), SegmentError> {
    download_files(&model_dir(models_dir, m), m.files, progress)
}

/// [`download`] for any pinned file list: missing files of `files` go into `dir`, each verified
/// against its SHA-256 and size before it is renamed into place.
#[cfg(feature = "download")]
pub fn download_files(dir: &Path, files: &[ModelFile], progress: DownloadProgress) -> Result<(), SegmentError> {
    use sha2::Digest;
    use std::io::{Read, Write};
    std::fs::create_dir_all(dir)?;
    let total: u64 = files.iter().filter(|f| !std::fs::metadata(dir.join(f.name)).is_ok_and(|md| md.len() == f.size)).map(|f| f.size).sum();
    let agent = agent();
    let mut done = 0u64;
    for f in files {
        let dest = dir.join(f.name);
        if std::fs::metadata(&dest).is_ok_and(|md| md.len() == f.size) {
            continue;
        }
        let part = dir.join(format!("{}.part", f.name));
        let resp = agent.get(f.url).call().map_err(|e| SegmentError::Download(format!("{}: {e}", f.url)))?;
        let mut body = resp.into_body();
        let mut reader = body.as_reader();
        let mut out = std::io::BufWriter::new(std::fs::File::create(&part)?);
        let mut hasher = sha2::Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| SegmentError::Download(format!("{}: {e}", f.name)))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            got += n as u64;
            if !progress(done + got, total, f.name) {
                drop(out);
                let _ = std::fs::remove_file(&part);
                return Err(SegmentError::Cancelled);
            }
        }
        out.flush()?;
        drop(out);
        let sha: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if sha != f.sha256 || got != f.size {
            let _ = std::fs::remove_file(&part);
            return Err(SegmentError::Download(format!("{}: checksum mismatch (got {sha}, {got} bytes; expected {}, {} bytes)", f.name, f.sha256, f.size)));
        }
        std::fs::rename(&part, &dest)?;
        done += f.size;
    }
    Ok(())
}

#[cfg(feature = "download")]
fn agent() -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_is_pinned_and_checksummed() {
        for m in catalogue() {
            assert!(m.size() > 10_000_000, "{}", m.id);
            assert!(for_subject(m.subject).is_some_and(|s| s.id == m.id));
            for f in m.files {
                assert!(f.url.starts_with("https://github.com/jamjamjon/assets/releases/download/"), "{}", f.url);
                assert!(f.url.ends_with(f.name));
                assert_eq!(f.sha256.len(), 64, "{}", f.name);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            }
        }
        assert_eq!(for_subject(Subject::Object).map(|m| m.id), Some("sam2-small"));
        assert_eq!(for_subject(Subject::Human).map(|m| m.id), Some("birefnet-hr-matting"));
    }

    #[test]
    fn installed_checks_sizes() {
        let dir = std::env::temp_dir().join(format!("filmcraft-segment-models-{}", std::process::id()));
        let m = find("sam2-small").unwrap();
        assert!(!installed(&dir, m));
        assert_eq!(missing_bytes(&dir, m), m.size());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
