//! Beacon HTTP server entry point.
//!
//! Loads a model and tokenizer at startup, then serves Ollama-compatible
//! and OpenAI-compatible REST APIs.
//!
//! ## Environment variables
//!
//! - `BEACON_MODEL` — path to `.gguf` or `.beacon` model file (required)
//! - `BEACON_TOKENIZER` — path to `tokenizer.json` (required)
//! - `BEACON_PORT` — port to listen on (default `11434`)

use std::path::Path;
use std::sync::{Arc, Mutex};

use beacon_server::AppState;

#[tokio::main]
async fn main() {
    let model_path = std::env::var("BEACON_MODEL").unwrap_or_else(|_| {
        eprintln!("error: BEACON_MODEL environment variable is required");
        eprintln!("  Set it to the path of a .gguf or .beacon model file.");
        std::process::exit(1);
    });

    let tokenizer_path = std::env::var("BEACON_TOKENIZER").unwrap_or_else(|_| {
        eprintln!("error: BEACON_TOKENIZER environment variable is required");
        eprintln!("  Set it to the path of a tokenizer.json file.");
        std::process::exit(1);
    });

    let port = std::env::var("BEACON_PORT").unwrap_or_else(|_| String::from("11434"));
    let addr = format!("0.0.0.0:{port}");

    // Derive a human-readable model name from the file path.
    let model_name = Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("beacon-model")
        .to_owned();

    eprintln!("Beacon server starting...");
    eprintln!("  Model     : {model_path}");
    eprintln!("  Tokenizer : {tokenizer_path}");
    eprintln!("  Port      : {port}");
    eprintln!();

    // Load model.
    eprintln!("Loading model...");
    let beacon_file = load_model(Path::new(&model_path));

    // Load tokenizer.
    eprintln!("Loading tokenizer...");
    let tokenizer = beacon_tokenizer::BeaconTokenizer::from_file(Path::new(&tokenizer_path))
        .unwrap_or_else(|e| {
            eprintln!("error: failed to load tokenizer: {e}");
            std::process::exit(1);
        });

    // Create MLX engine.
    eprintln!("Creating MLX engine...");
    let ctx = Arc::new(beacon_mlx::MlxContext::new().unwrap_or_else(|e| {
        eprintln!("error: failed to create MLX context: {e}");
        std::process::exit(1);
    }));
    let backend = beacon_core::MlxBackend::new(Arc::clone(&ctx));
    let engine = beacon_core::Engine::load(&beacon_file, backend).unwrap_or_else(|e| {
        eprintln!("error: failed to load engine: {e}");
        std::process::exit(1);
    });

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer,
        model_name: model_name.clone(),
    });

    eprintln!();
    eprintln!("Model loaded: {model_name}");
    eprintln!("  Architecture : {:?}", beacon_file.config.architecture);
    eprintln!("  Layers       : {}", beacon_file.config.num_layers);
    eprintln!("  Vocab size   : {}", beacon_file.config.vocab_size);
    eprintln!();
    eprintln!("Endpoints:");
    eprintln!("  Ollama  : POST /api/generate, POST /api/chat, GET /api/tags");
    eprintln!("  OpenAI  : POST /v1/chat/completions, GET /v1/models");
    eprintln!();
    eprintln!("Beacon server listening on {addr}");

    let app = beacon_server::create_router(state);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind TCP listener");

    axum::serve(listener, app)
        .await
        .expect("server exited with error");
}

/// Load a `.gguf` or `.beacon` model file, performing GGUF-to-beacon
/// conversion if needed.
fn load_model(model_path: &Path) -> beacon_format::BeaconFile {
    let ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "gguf" => {
            eprintln!("  Converting GGUF to .beacon format...");
            beacon_format::load_or_convert(model_path, None).unwrap_or_else(|e| {
                eprintln!("error: failed to convert GGUF: {e}");
                std::process::exit(1);
            })
        }
        "beacon" => beacon_format::BeaconFile::open(model_path).unwrap_or_else(|e| {
            eprintln!("error: failed to open .beacon file: {e}");
            std::process::exit(1);
        }),
        _ => {
            eprintln!("error: unsupported file extension: .{ext}");
            eprintln!("  Beacon supports .gguf and .beacon files.");
            std::process::exit(1);
        }
    }
}
