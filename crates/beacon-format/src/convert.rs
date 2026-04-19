//! GGUF → `.beacon` conversion and caching.
//!
//! During conversion, all quantized tensors (`Q4_0`, `Q4_K`, `Q8_0`, etc.) are
//! dequantized to F16 so that the resulting `.beacon` file contains only
//! non-quantized data. This makes subsequent loads zero-copy — no
//! dequantization needed at model load time.
//!
//! Dequantization of individual tensors is parallelized via rayon.

use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::beacon::reader::BeaconFile;
use crate::beacon::writer;
use crate::dequant;
use crate::error::FormatError;
use crate::gguf::GgufFile;
use crate::tensor_meta::TensorMeta;
use crate::BeaconDtype;

/// Convert a GGUF file to `.beacon` format.
///
/// Reads the GGUF, extracts model config and tokenizer metadata, dequantizes
/// all quantized tensors to F16 in parallel, and writes a self-contained
/// `.beacon` file with 64-byte-aligned tensor data.
pub fn convert_gguf_to_beacon(
    gguf_path: &Path,
    beacon_path: &Path,
) -> Result<BeaconFile, FormatError> {
    let gguf = GgufFile::open(gguf_path)?;

    let config = gguf.model_config()?;
    let tokenizer_json = gguf.tokenizer_json()?;

    // Collect raw tensor data slices from the GGUF mmap.
    let raw_data: Vec<&[u8]> = gguf.tensors.iter().map(|t| gguf.tensor_data(t)).collect();

    // Identify which tensors are quantized and need dequantization.
    // Parallel dequantization: process all quantized tensors concurrently.
    let dequanted: Vec<Option<Vec<u8>>> = gguf
        .tensors
        .par_iter()
        .zip(raw_data.par_iter())
        .map(|(meta, data)| {
            if dequant::is_quantized(meta.dtype) {
                let f16_vals = dequant::dequantize_to_f16(data, meta.dtype, meta.num_elements());
                // Convert Vec<u16> to Vec<u8> (little-endian).
                let mut bytes = vec![0u8; f16_vals.len() * 2];
                for (i, &val) in f16_vals.iter().enumerate() {
                    let le = val.to_le_bytes();
                    bytes[i * 2] = le[0];
                    bytes[i * 2 + 1] = le[1];
                }
                Some(bytes)
            } else {
                None
            }
        })
        .collect();

    // Build updated tensor metadata and data slices. For quantized tensors,
    // replace dtype with F16 and use the dequanted bytes.
    let mut out_tensors = Vec::with_capacity(gguf.tensors.len());
    let mut owned_data: Vec<Vec<u8>> = Vec::with_capacity(gguf.tensors.len());

    for (i, meta) in gguf.tensors.iter().enumerate() {
        if let Some(ref f16_bytes) = dequanted[i] {
            // Quantized tensor → dequanted to F16.
            let num_elems = meta.num_elements();
            let f16_data_len = num_elems * 2; // 2 bytes per F16 element
            out_tensors.push(TensorMeta {
                name: meta.name.clone(),
                dtype: BeaconDtype::F16,
                shape: meta.shape.clone(),
                data_offset: 0, // recomputed by writer
                data_length: f16_data_len,
            });
            owned_data.push(f16_bytes.clone());
        } else {
            // Non-quantized tensor → pass through unchanged.
            out_tensors.push(meta.clone());
            owned_data.push(raw_data[i].to_vec());
        }
    }

    let tensor_data_slices: Vec<&[u8]> = owned_data.iter().map(Vec::as_slice).collect();

    // Ensure parent directory exists.
    if let Some(parent) = beacon_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    writer::write_beacon(
        beacon_path,
        &config,
        &tokenizer_json,
        &out_tensors,
        &tensor_data_slices,
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
