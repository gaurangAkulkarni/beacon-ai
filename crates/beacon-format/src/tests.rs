//! Tests for beacon-format: synthetic GGUF parsing, `.beacon` round-trip,
//! and an ignored test for real GGUF files.

use crate::beacon::BeaconFile;
use crate::config::{Architecture, ModelConfig};
use crate::convert::convert_gguf_to_beacon;
use crate::dtype::BeaconDtype;
use crate::gguf::GgufFile;

// ---------------------------------------------------------------------------
// Synthetic GGUF builder
// ---------------------------------------------------------------------------

/// Build a minimal valid GGUF v3 file in memory.
///
/// Contains two tiny F32 tensors and enough metadata to extract a
/// `ModelConfig` for the "llama" architecture.
fn build_test_gguf() -> Vec<u8> {
    let mut buf = Vec::new();

    // --- Fixed header (24 bytes) ---
    buf.extend_from_slice(b"GGUF"); // magic
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
                                                // We'll write metadata count after building the KV pairs.
    let meta_count_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder

    // --- Metadata KV pairs ---
    let mut kv_count: u64 = 0;

    gguf_write_string_kv(&mut buf, "general.architecture", "llama");
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "llama.embedding_length", 64);
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "llama.block_count", 2);
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "llama.attention.head_count", 4);
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "llama.attention.head_count_kv", 2);
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "llama.feed_forward_length", 128);
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "llama.context_length", 512);
    kv_count += 1;
    gguf_write_f32_kv(&mut buf, "llama.rope.freq_base", 10000.0);
    kv_count += 1;
    gguf_write_f32_kv(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "tokenizer.ggml.bos_token_id", 1);
    kv_count += 1;
    gguf_write_u32_kv(&mut buf, "tokenizer.ggml.eos_token_id", 2);
    kv_count += 1;

    // Patch metadata count.
    buf[meta_count_pos..meta_count_pos + 8].copy_from_slice(&kv_count.to_le_bytes());

    // --- Tensor info entries ---
    // Tensor 0: "weight_a", shape [2, 3], F32 (type 0), offset 0
    write_tensor_info(&mut buf, "weight_a", &[2, 3], 0, 0);

    // Tensor 1: "weight_b", shape [3, 2], F32 (type 0), offset = 2*3*4 = 24
    write_tensor_info(&mut buf, "weight_b", &[3, 2], 0, 24);

    // --- Alignment padding to default alignment (32 bytes) ---
    let align = 32u64;
    let current = buf.len() as u64;
    let padded = current.div_ceil(align) * align;
    #[allow(clippy::cast_possible_truncation)]
    buf.resize(padded as usize, 0);

    // --- Tensor data ---
    // weight_a: 6 f32 values = 24 bytes
    let a_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for v in &a_data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    // weight_b: 6 f32 values = 24 bytes
    let b_data: [f32; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    for v in &b_data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

fn gguf_write_string_kv(buf: &mut Vec<u8>, key: &str, val: &str) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val.as_bytes());
}

fn gguf_write_u32_kv(buf: &mut Vec<u8>, key: &str, val: u32) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes()); // type: uint32
    buf.extend_from_slice(&val.to_le_bytes());
}

fn gguf_write_f32_kv(buf: &mut Vec<u8>, key: &str, val: f32) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&6u32.to_le_bytes()); // type: float32
    buf.extend_from_slice(&val.to_le_bytes());
}

fn write_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64], gguf_type: u32, offset: u64) {
    // name: u64 len + bytes
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    // ndim: u32
    #[allow(clippy::cast_possible_truncation)]
    let ndim = shape.len() as u32;
    buf.extend_from_slice(&ndim.to_le_bytes());
    // dims: u64 each
    for &d in shape {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    // type: u32
    buf.extend_from_slice(&gguf_type.to_le_bytes());
    // offset: u64 (relative to data section start)
    buf.extend_from_slice(&offset.to_le_bytes());
}

// ---------------------------------------------------------------------------
// GGUF parser tests
// ---------------------------------------------------------------------------

#[test]
fn parse_synthetic_gguf() {
    let data = build_test_gguf();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &data).unwrap();

    let gguf = GgufFile::open(tmp.path()).unwrap();

    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.tensors.len(), 2);

    // Check tensor metadata.
    assert_eq!(gguf.tensors[0].name, "weight_a");
    assert_eq!(gguf.tensors[0].shape, vec![2, 3]);
    assert_eq!(gguf.tensors[0].dtype, BeaconDtype::F32);
    assert_eq!(gguf.tensors[0].data_length, 24); // 6 * 4

    assert_eq!(gguf.tensors[1].name, "weight_b");
    assert_eq!(gguf.tensors[1].shape, vec![3, 2]);
    assert_eq!(gguf.tensors[1].dtype, BeaconDtype::F32);
    assert_eq!(gguf.tensors[1].data_length, 24);

    // Check tensor data.
    let a_bytes = gguf.tensor_data(&gguf.tensors[0]);
    let a_vals: Vec<f32> = a_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(a_vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let b_bytes = gguf.tensor_data(&gguf.tensors[1]);
    let b_vals: Vec<f32> = b_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(b_vals, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
}

#[test]
fn extract_model_config_from_gguf() {
    let data = build_test_gguf();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &data).unwrap();

    let gguf = GgufFile::open(tmp.path()).unwrap();
    let config = gguf.model_config().unwrap();

    assert_eq!(config.architecture, Architecture::Llama);
    assert_eq!(config.hidden_size, 64);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.intermediate_size, 128);
    assert_eq!(config.head_dim, 16); // 64 / 4
    assert_eq!(config.max_position_embeddings, 512);
    assert!((config.rope_theta - 10000.0).abs() < 1e-3);
    assert!((config.rms_norm_eps - 1e-5).abs() < 1e-8);
    assert_eq!(config.bos_token_id, Some(1));
    assert_eq!(config.eos_token_ids, vec![2]);
}

// ---------------------------------------------------------------------------
// .beacon round-trip test
// ---------------------------------------------------------------------------

#[test]
fn beacon_round_trip() {
    let config = ModelConfig {
        architecture: Architecture::Qwen,
        hidden_size: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        head_dim: 16,
        vocab_size: 1000,
        max_position_embeddings: 512,
        rope_theta: 10000.0,
        rope_scaling: None,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        bos_token_id: Some(1),
        eos_token_ids: vec![2],
        chat_template: None,
    };

    let tokenizer_json = r#"{"vocab_size":1000}"#;

    // Create some tensor data.
    let tensor_a_data: Vec<u8> = (0..24).collect(); // 24 bytes
    let tensors = vec![
        crate::TensorMeta {
            name: "layer.0.weight".to_owned(),
            dtype: BeaconDtype::F32,
            shape: vec![2, 3],
            data_offset: 0, // will be recomputed by writer
            data_length: 24,
        },
        crate::TensorMeta {
            name: "layer.1.weight".to_owned(),
            dtype: BeaconDtype::Q4K,
            shape: vec![256],
            data_offset: 0,
            data_length: 144,
        },
    ];

    // For the Q4K tensor, create 144 bytes of data.
    let tensor_b_q4k: Vec<u8> = (0..144u8).collect();

    let data_map: std::collections::HashMap<&str, &[u8]> = [
        ("layer.0.weight", tensor_a_data.as_slice()),
        ("layer.1.weight", tensor_b_q4k.as_slice()),
    ]
    .into_iter()
    .collect();

    let tmp_dir = tempfile::tempdir().unwrap();
    let beacon_path = tmp_dir.path().join("test.beacon");

    let tensor_data_slices: Vec<&[u8]> = tensors
        .iter()
        .map(|t| *data_map.get(t.name.as_str()).unwrap())
        .collect();
    crate::beacon::writer::write_beacon(
        &beacon_path,
        &config,
        tokenizer_json,
        &tensors,
        &tensor_data_slices,
    )
    .unwrap();

    // Read back.
    let loaded = BeaconFile::open(&beacon_path).unwrap();

    assert_eq!(loaded.config, config);
    assert_eq!(loaded.tokenizer_json, tokenizer_json);
    assert_eq!(loaded.tensors.len(), 2);
    assert_eq!(loaded.tensors[0].name, "layer.0.weight");
    assert_eq!(loaded.tensors[0].dtype, BeaconDtype::F32);
    assert_eq!(loaded.tensors[0].shape, vec![2, 3]);
    assert_eq!(loaded.tensors[0].data_length, 24);
    assert_eq!(loaded.tensors[1].name, "layer.1.weight");
    assert_eq!(loaded.tensors[1].dtype, BeaconDtype::Q4K);

    // Verify tensor data.
    let a_readback = loaded.tensor_data(&loaded.tensors[0]);
    assert_eq!(a_readback, tensor_a_data.as_slice());

    let b_readback = loaded.tensor_data(&loaded.tensors[1]);
    assert_eq!(b_readback, tensor_b_q4k.as_slice());

    // Verify 64-byte alignment of tensor data offsets.
    for t in &loaded.tensors {
        assert_eq!(
            t.data_offset % 64,
            0,
            "tensor {} data_offset {} is not 64-byte aligned",
            t.name,
            t.data_offset
        );
    }
}

// ---------------------------------------------------------------------------
// Full GGUF → .beacon conversion test
// ---------------------------------------------------------------------------

#[test]
fn convert_synthetic_gguf_to_beacon() {
    let data = build_test_gguf();
    let tmp_dir = tempfile::tempdir().unwrap();
    let gguf_path = tmp_dir.path().join("test.gguf");
    let beacon_path = tmp_dir.path().join("test.beacon");

    std::fs::write(&gguf_path, &data).unwrap();

    let beacon = convert_gguf_to_beacon(&gguf_path, &beacon_path).unwrap();

    assert_eq!(beacon.config.architecture, Architecture::Llama);
    assert_eq!(beacon.tensors.len(), 2);
    assert_eq!(beacon.tensors[0].name, "weight_a");
    assert_eq!(beacon.tensors[1].name, "weight_b");

    // Verify tensor data survived the conversion.
    let a_vals: Vec<f32> = beacon
        .tensor_data(&beacon.tensors[0])
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(a_vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let b_vals: Vec<f32> = beacon
        .tensor_data(&beacon.tensors[1])
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(b_vals, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
}

// ---------------------------------------------------------------------------
// Config JSON round-trip
// ---------------------------------------------------------------------------

#[test]
fn model_config_json_round_trip() {
    let config = ModelConfig {
        architecture: Architecture::Gemma,
        hidden_size: 2048,
        num_layers: 18,
        num_heads: 8,
        num_kv_heads: 1,
        intermediate_size: 16384,
        head_dim: 256,
        vocab_size: 256_000,
        max_position_embeddings: 8192,
        rope_theta: 10000.0,
        rope_scaling: Some(crate::RopeScaling {
            type_: "linear".to_owned(),
            factor: 4.0,
        }),
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        bos_token_id: Some(2),
        eos_token_ids: vec![1, 107],
        chat_template: Some("<|im_start|>".to_owned()),
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config, deserialized);
}

// ---------------------------------------------------------------------------
// dtype tests
// ---------------------------------------------------------------------------

#[test]
fn beacon_dtype_round_trip() {
    let dtypes = [
        BeaconDtype::F32,
        BeaconDtype::F16,
        BeaconDtype::BF16,
        BeaconDtype::I32,
        BeaconDtype::I8,
        BeaconDtype::Q4_0,
        BeaconDtype::Q4K,
        BeaconDtype::Q5K,
        BeaconDtype::Q6K,
        BeaconDtype::Q8_0,
    ];
    for dt in &dtypes {
        let v = *dt as u32;
        let back = BeaconDtype::from_u32(v).unwrap();
        assert_eq!(*dt, back);
    }
}

#[test]
fn beacon_dtype_data_length() {
    // F32: 6 elements = 24 bytes
    assert_eq!(BeaconDtype::F32.data_byte_length(6), 24);
    // Q4_0: block_size=32, bytes_per_block=18; 64 elements = 2 blocks = 36 bytes
    assert_eq!(BeaconDtype::Q4_0.data_byte_length(64), 36);
    // Q4K: block_size=256, bytes_per_block=144; 256 elements = 1 block = 144 bytes
    assert_eq!(BeaconDtype::Q4K.data_byte_length(256), 144);
}

// ---------------------------------------------------------------------------
// Real model test (ignored by default)
// ---------------------------------------------------------------------------

/// Load a real GGUF file, convert to `.beacon`, and enumerate tensors.
///
/// Run with: `BEACON_TEST_GGUF=/path/to/model.gguf cargo test -p beacon-format -- --ignored`
#[test]
#[ignore = "requires BEACON_TEST_GGUF env var"]
fn convert_real_gguf() {
    let gguf_path = std::env::var("BEACON_TEST_GGUF")
        .expect("set BEACON_TEST_GGUF=/path/to/model.gguf to run this test");

    let tmp_dir = tempfile::tempdir().unwrap();
    let beacon_path = tmp_dir.path().join("model.beacon");

    eprintln!("Converting {gguf_path} → {}", beacon_path.display());

    let beacon = convert_gguf_to_beacon(std::path::Path::new(&gguf_path), &beacon_path).unwrap();

    eprintln!("Architecture: {:?}", beacon.config.architecture);
    eprintln!("Layers: {}", beacon.config.num_layers);
    eprintln!("Hidden size: {}", beacon.config.hidden_size);
    eprintln!("Tensors: {}", beacon.tensors.len());

    for t in &beacon.tensors {
        eprintln!(
            "  {} {:?} {:?} ({} bytes @ offset {})",
            t.name, t.dtype, t.shape, t.data_length, t.data_offset
        );
    }

    assert!(!beacon.tensors.is_empty(), "expected at least one tensor");
    assert!(beacon.config.num_layers > 0, "expected at least one layer");
}
