//! HTTP download from `HuggingFace` Hub with progress reporting.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};

use crate::error::RegistryError;
use crate::models::ModelSpec;

/// Result of a successful `pull_model` operation, containing paths to all
/// downloaded and converted files.
#[derive(Debug)]
pub struct PullResult {
    /// Path to the downloaded GGUF file.
    pub gguf_path: PathBuf,
    /// Path to the downloaded `tokenizer.json`.
    pub tokenizer_path: PathBuf,
    /// Path to the converted `.beacon` file.
    pub beacon_path: PathBuf,
    /// Display name of the model.
    pub model_name: String,
}

/// Download a model from `HuggingFace` Hub and convert it to `.beacon` format.
///
/// The flow is:
/// 1. Create the cache directory (`cache_dir/{display_name}/`).
/// 2. Download the GGUF file (skip if already present).
/// 3. Download `tokenizer.json` (skip if already present).
/// 4. Convert GGUF to `.beacon` (skip if already present).
///
/// Returns paths to all downloaded/converted files.
pub fn pull_model(spec: &ModelSpec, cache_dir: &Path) -> Result<PullResult, RegistryError> {
    let model_dir = cache_dir.join(&spec.display_name);
    std::fs::create_dir_all(&model_dir)?;

    let gguf_path = model_dir.join("model.gguf");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let beacon_path = model_dir.join("model.beacon");

    // Download GGUF.
    if gguf_path.exists() {
        eprintln!("  GGUF already cached: {}", gguf_path.display());
    } else {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, spec.gguf_file
        );
        download_file(&url, &gguf_path, &format!("Downloading {}", spec.gguf_file))?;
    }

    // Download tokenizer.json.
    if tokenizer_path.exists() {
        eprintln!("  Tokenizer already cached: {}", tokenizer_path.display());
    } else {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            spec.tokenizer_repo
        );
        download_file(&url, &tokenizer_path, "Downloading tokenizer.json")?;
    }

    // Convert GGUF to .beacon.
    if beacon_path.exists() {
        eprintln!("  .beacon already cached: {}", beacon_path.display());
    } else {
        eprintln!("  Converting GGUF to .beacon format...");
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .expect("valid spinner template"),
        );
        pb.set_message("Converting...");
        pb.enable_steady_tick(std::time::Duration::from_millis(80));

        beacon_format::convert_gguf_to_beacon(&gguf_path, &beacon_path)?;

        pb.finish_with_message("Conversion complete.");
    }

    Ok(PullResult {
        gguf_path,
        tokenizer_path,
        beacon_path,
        model_name: spec.display_name.clone(),
    })
}

/// Download a single file from `url` to `dest`, showing a progress bar.
///
/// Uses a temporary file alongside the destination, then renames atomically
/// on success. This prevents partial downloads from being treated as complete.
#[allow(clippy::similar_names)]
fn download_file(url: &str, dest: &Path, desc: &str) -> Result<(), RegistryError> {
    let resp = reqwest::blocking::get(url).map_err(|e| RegistryError::Download(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(RegistryError::Download(format!(
            "HTTP {} for {}",
            resp.status(),
            url
        )));
    }

    let total_size = resp.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n  [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("valid progress template")
            .progress_chars("##-"),
    );
    pb.set_message(desc.to_owned());

    // Write to a temp file next to the destination, then rename.
    let tmp_path = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp_path).map_err(RegistryError::Io)?;

    let mut reader = resp;
    let mut buf = [0u8; 8192];
    let mut downloaded: u64 = 0;

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| RegistryError::Download(format!("read error: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    pb.finish();

    // Flush and rename to final destination.
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp_path, dest)?;

    Ok(())
}

/// Return the default models cache directory: `~/.beacon/models/`.
pub fn default_models_dir() -> PathBuf {
    let base = std::env::var("BEACON_CACHE_DIR").map_or_else(
        |_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".to_owned());
            PathBuf::from(home).join(".beacon")
        },
        PathBuf::from,
    );
    base.join("models")
}
