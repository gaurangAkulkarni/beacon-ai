//! GGUF → `.beacon` conversion and caching.
//!
//! Converts GGUF files to the `.beacon` format, preserving tensor data as-is
//! (including quantized types). Dequantization happens at engine load time
//! via the C shim's gguflib-based dequantizer, which is battle-tested and
//! handles all GGUF quant types correctly.

use std::path::{Path, PathBuf};

use crate::beacon::reader::BeaconFile;
use crate::beacon::writer;
use crate::error::FormatError;
use crate::gguf::GgufFile;

/// Convert a GGUF file to `.beacon` format.
///
/// Reads the GGUF, extracts model config and tokenizer metadata, and writes
/// a self-contained `.beacon` file with 64-byte-aligned tensor data. Tensor
/// data (including quantized types) is copied as-is without dequantization.
pub fn convert_gguf_to_beacon(
    gguf_path: &Path,
    beacon_path: &Path,
) -> Result<BeaconFile, FormatError> {
    let gguf = GgufFile::open(gguf_path)?;

    let config = gguf.model_config()?;
    let tokenizer_json = gguf.tokenizer_json()?;

    // Collect raw tensor data slices from the GGUF mmap (zero-copy references).
    let tensor_data: Vec<&[u8]> = gguf.tensors.iter().map(|t| gguf.tensor_data(t)).collect();

    // Ensure parent directory exists.
    if let Some(parent) = beacon_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    writer::write_beacon(
        beacon_path,
        &config,
        &tokenizer_json,
        &gguf.tensors,
        &tensor_data,
    )?;

    BeaconFile::open(beacon_path)
}

/// Load a `.beacon` file, converting from GGUF on first access.
///
/// If `cache_dir` is `None`, defaults to `~/.beacon/models/`. The `.beacon`
/// file is cached using the GGUF file's stem as the directory name.
pub fn load_or_convert(
    gguf_path: &Path,
    cache_dir: Option<&Path>,
) -> Result<BeaconFile, FormatError> {
    let beacon_path = cached_beacon_path(gguf_path, cache_dir);

    // If a cached .beacon already exists, use it.
    if beacon_path.exists() {
        return BeaconFile::open(&beacon_path);
    }

    convert_gguf_to_beacon(gguf_path, &beacon_path)
}

/// Compute the cache path for a given GGUF file.
fn cached_beacon_path(gguf_path: &Path, cache_dir: Option<&Path>) -> PathBuf {
    let model_name = gguf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let base = match cache_dir {
        Some(dir) => dir.to_path_buf(),
        None => default_cache_dir(),
    };

    base.join(model_name).join("model.beacon")
}

/// Default cache directory: `~/.beacon/models/`.
fn default_cache_dir() -> PathBuf {
    dirs_fallback().join("models")
}

/// Best-effort home directory resolution without adding a dependency.
fn dirs_fallback() -> PathBuf {
    std::env::var("BEACON_CACHE_DIR").map_or_else(
        |_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".to_owned());
            PathBuf::from(home).join(".beacon")
        },
        PathBuf::from,
    )
}
