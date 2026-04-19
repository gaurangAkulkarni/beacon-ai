//! Beacon CLI entry point.
//!
//! Binary named `beacon`. Subcommands: `pull`, `run`, `list`, `remove`, `info`,
//! `serve` per architecture doc section 12.3.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use beacon_core::ComputeBackend as _;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;

// ---------------------------------------------------------------------------
// CLI structure
// ---------------------------------------------------------------------------

/// `MLX`-first LLM inference for Apple Silicon.
#[derive(Parser)]
#[command(name = "beacon", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download a model from Hugging Face Hub and convert to `.beacon` format.
    Pull {
        /// Model identifier (e.g. `Qwen/Qwen2.5-3B-Instruct-GGUF`).
        model: String,
    },
    /// Run inference on a model with a given prompt.
    Run {
        /// Path to a `.gguf` or `.beacon` model file, or a model name.
        model: String,
        /// The prompt to send to the model.
        prompt: String,
        /// Maximum number of tokens to generate.
        #[arg(long, default_value = "512")]
        max_tokens: usize,
        /// Sampling temperature (0.0 = greedy).
        #[arg(long, default_value = "0.0")]
        temperature: f32,
        /// Top-k sampling (omit for no top-k filtering).
        #[arg(long)]
        top_k: Option<usize>,
        /// Top-p / nucleus sampling (omit for no top-p filtering).
        #[arg(long)]
        top_p: Option<f32>,
        /// Path to tokenizer.json (auto-detected if not specified).
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// List downloaded models in the local cache.
    List,
    /// Remove a downloaded model from the local cache.
    Remove {
        /// Model name (directory name under `~/.beacon/models/`).
        model: String,
    },
    /// Show model information from a `.beacon` or `.gguf` file.
    Info {
        /// Path to a `.gguf` or `.beacon` model file.
        model: String,
    },
    /// Start the HTTP server (placeholder for Step 11).
    Serve {
        /// Port to listen on.
        #[arg(long, default_value = "11434")]
        port: u16,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Pull { model } => {
            cmd_pull(&model);
            Ok(())
        }
        Command::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            tokenizer,
        } => cmd_run(
            &model,
            &prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            tokenizer.as_deref(),
        ),
        Command::List => cmd_list(),
        Command::Remove { model } => cmd_remove(&model),
        Command::Info { model } => cmd_info(&model),
        Command::Serve { port } => {
            cmd_serve(port);
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

/// `beacon pull` — placeholder until `beacon-registry` is implemented.
fn cmd_pull(model: &str) {
    eprintln!(
        "{} {} is not yet implemented (beacon-registry lands in a future step).",
        "note:".yellow().bold(),
        "beacon pull".bold()
    );
    eprintln!();
    eprintln!("  Model requested: {}", model.cyan());
    eprintln!();
    eprintln!("  To use a GGUF model you already have on disk:");
    eprintln!(
        "    {} /path/to/model.gguf \"your prompt\"",
        "beacon run".green()
    );
}

/// Locate the `tokenizer.json` file — either from an explicit path or by
/// looking next to the model file.
fn find_tokenizer(model_path: &Path, explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        bail!("Tokenizer file not found: {p}");
    }

    // Check next to the model file.
    if let Some(dir) = model_path.parent() {
        let candidate = dir.join("tokenizer.json");
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    bail!(
        "No tokenizer.json found next to {}.\n\
         Use --tokenizer /path/to/tokenizer.json to specify one.",
        model_path.display()
    );
}

/// Load a `.gguf` or `.beacon` model file, performing GGUF-to-beacon
/// conversion if needed.
fn load_model(model_path: &Path) -> Result<beacon_format::BeaconFile> {
    let ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "gguf" => {
            eprintln!(
                "{} Converting GGUF to .beacon format...",
                "=>".green().bold()
            );
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .expect("valid template"),
            );
            pb.set_message("Reading GGUF and writing .beacon cache...");
            pb.enable_steady_tick(std::time::Duration::from_millis(80));

            let bf = beacon_format::load_or_convert(model_path, None)
                .context("failed to convert GGUF to .beacon")?;

            pb.finish_with_message("Conversion complete.");
            Ok(bf)
        }
        "beacon" => {
            eprintln!("{} Loading .beacon file...", "=>".green().bold());
            beacon_format::BeaconFile::open(model_path).context("failed to open .beacon file")
        }
        _ => {
            bail!(
                "Unsupported file extension: .{ext}\n\
                 Beacon supports .gguf and .beacon files."
            );
        }
    }
}

/// Print model info and generation parameters to stderr.
fn print_run_header(
    beacon_file: &beacon_format::BeaconFile,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) {
    let cfg = &beacon_file.config;
    eprintln!();
    eprintln!("{}", "Model info".bold().underline());
    eprintln!("  Architecture : {:?}", cfg.architecture);
    eprintln!("  Hidden size  : {}", cfg.hidden_size);
    eprintln!("  Layers       : {}", cfg.num_layers);
    eprintln!("  Heads        : {}", cfg.num_heads);
    eprintln!("  KV heads     : {}", cfg.num_kv_heads);
    eprintln!("  Vocab size   : {}", cfg.vocab_size);
    eprintln!("  Tensors      : {}", beacon_file.tensors.len());
    eprintln!();

    eprintln!("{}", "Generation parameters".bold().underline());
    eprintln!("  Prompt       : {}", prompt.cyan());
    eprintln!("  Max tokens   : {max_tokens}");
    eprintln!("  Temperature  : {temperature}");
    if let Some(k) = top_k {
        eprintln!("  Top-k        : {k}");
    }
    if let Some(p) = top_p {
        eprintln!("  Top-p        : {p}");
    }
    eprintln!();
}

/// Run the auto-regressive generation loop, printing tokens to stdout.
fn generate(
    engine: &mut beacon_core::Engine<beacon_core::MlxBackend>,
    tokenizer: &beacon_tokenizer::BeaconTokenizer,
    token_ids: &[u32],
    params: &beacon_scheduler::GenerationParams,
    eos_token_ids: &[u32],
    vocab_size: usize,
    max_tokens: usize,
) -> Result<()> {
    let mut rng = rand::rng();

    // Prefill: forward the entire prompt at once.
    let logits = engine
        .forward(token_ids, 0)
        .context("prefill forward pass failed")?;

    // The logits tensor is [seq_len, vocab_size]. Read all, take the last row.
    let all_logits = engine
        .backend_ref()
        .read_f32(&logits, token_ids.len() * vocab_size)
        .context("failed to read prefill logits")?;

    let last_row_start = (token_ids.len() - 1) * vocab_size;
    let mut logit_values = all_logits[last_row_start..last_row_start + vocab_size].to_vec();

    let mut prev_tokens = token_ids.to_vec();
    let mut position = token_ids.len();

    // Sample first generated token.
    let mut next_token =
        beacon_scheduler::sampling::sample(&mut logit_values, &prev_tokens, params, &mut rng);

    if eos_token_ids.contains(&next_token) {
        println!();
        return Ok(());
    }

    if let Ok(text) = tokenizer.decode(&[next_token], true) {
        print!("{text}");
        std::io::stdout().flush()?;
    }

    // Auto-regressive decode loop.
    for _ in 1..max_tokens {
        let logits = engine
            .forward(&[next_token], position)
            .context("decode forward pass failed")?;
        position += 1;

        let mut logit_vals = engine
            .backend_ref()
            .read_f32(&logits, vocab_size)
            .context("failed to read decode logits")?;

        prev_tokens.push(next_token);
        next_token =
            beacon_scheduler::sampling::sample(&mut logit_vals, &prev_tokens, params, &mut rng);

        if eos_token_ids.contains(&next_token) {
            break;
        }

        if let Ok(text) = tokenizer.decode(&[next_token], true) {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }

    println!(); // Final newline.
    Ok(())
}

/// `beacon run` — load a model and run end-to-end text generation.
#[allow(clippy::too_many_arguments)]
fn cmd_run(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    tokenizer_path: Option<&str>,
) -> Result<()> {
    let model_path = Path::new(model);

    if !model_path.exists() {
        bail!(
            "Model file not found: {}\n\n\
             Hint: provide a path to a .gguf or .beacon file, e.g.:\n  \
             beacon run /path/to/model.gguf \"Hello\"",
            model_path.display()
        );
    }

    let beacon_file = load_model(model_path)?;
    print_run_header(&beacon_file, prompt, max_tokens, temperature, top_k, top_p);

    // Find and load tokenizer.
    let tok_path = find_tokenizer(model_path, tokenizer_path)?;
    eprintln!(
        "{} Loading tokenizer from {}",
        "=>".green().bold(),
        tok_path.display()
    );
    let tokenizer = beacon_tokenizer::BeaconTokenizer::from_file(&tok_path)
        .context("failed to load tokenizer")?;

    // Create MLX backend + engine.
    eprintln!("{} Loading model weights...", "=>".green().bold());
    let ctx = Arc::new(beacon_mlx::MlxContext::new().context("failed to create MLX context")?);
    let backend = beacon_core::MlxBackend::new(Arc::clone(&ctx));
    let mut engine =
        beacon_core::Engine::load(&beacon_file, backend).context("failed to load engine")?;

    // Encode prompt.
    let token_ids = tokenizer
        .encode(prompt, false)
        .context("failed to encode prompt")?;
    eprintln!("  Prompt tokens: {}", token_ids.len());
    eprintln!();

    // Build sampling parameters.
    let cfg = &beacon_file.config;
    let params = beacon_scheduler::GenerationParams {
        max_tokens,
        temperature,
        top_k,
        top_p,
        stop_tokens: cfg.eos_token_ids.clone(),
        ..beacon_scheduler::GenerationParams::default()
    };

    generate(
        &mut engine,
        &tokenizer,
        &token_ids,
        &params,
        &cfg.eos_token_ids,
        cfg.vocab_size,
        max_tokens,
    )
}

/// `beacon list` — enumerate `.beacon` files in the local cache.
fn cmd_list() -> Result<()> {
    let cache_dir = default_models_dir();

    if !cache_dir.exists() {
        eprintln!("{} No models found.", "note:".yellow().bold());
        eprintln!(
            "  Cache directory does not exist: {}",
            cache_dir.display().to_string().dimmed()
        );
        return Ok(());
    }

    let mut found = false;
    let entries = std::fs::read_dir(&cache_dir).context("failed to read model cache directory")?;

    let mut table = String::new();

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let beacon_path = path.join("model.beacon");
        if !beacon_path.exists() {
            continue;
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>");
        let meta = std::fs::metadata(&beacon_path)?;

        #[allow(clippy::cast_precision_loss)]
        let size_mb = meta.len() as f64 / (1024.0 * 1024.0);

        if !found {
            eprintln!("{}", "Downloaded models".bold().underline());
            eprintln!();
            found = true;
        }

        writeln!(table, "  {:<40} {:>10.1} MB", name.green(), size_mb)?;
    }

    if found {
        eprint!("{table}");
    } else {
        eprintln!("{} No models found.", "note:".yellow().bold());
        eprintln!(
            "  Cache directory: {}",
            cache_dir.display().to_string().dimmed()
        );
    }

    Ok(())
}

/// `beacon remove` — delete a model from the local cache.
fn cmd_remove(model: &str) -> Result<()> {
    let model_dir = default_models_dir().join(model);

    if !model_dir.exists() {
        bail!(
            "Model '{}' not found in cache directory: {}",
            model,
            model_dir.display()
        );
    }

    std::fs::remove_dir_all(&model_dir)
        .with_context(|| format!("failed to remove {}", model_dir.display()))?;

    eprintln!("{} Removed model '{}'.", "ok:".green().bold(), model.cyan());
    Ok(())
}

/// `beacon info` — show model config and tensor summary.
fn cmd_info(model: &str) -> Result<()> {
    let model_path = Path::new(model);

    if !model_path.exists() {
        bail!("File not found: {}", model_path.display());
    }

    let ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "beacon" => info_beacon(model_path),
        "gguf" => info_gguf(model_path),
        _ => bail!(
            "Unsupported file extension: .{ext}\n\
             Beacon supports .gguf and .beacon files."
        ),
    }
}

/// Print info from a `.beacon` file.
fn info_beacon(path: &Path) -> Result<()> {
    let bf = beacon_format::BeaconFile::open(path).context("failed to open .beacon file")?;

    let cfg = &bf.config;
    eprintln!("{}", "Beacon model file".bold().underline());
    eprintln!("  Path         : {}", path.display());
    eprintln!("  Architecture : {:?}", cfg.architecture);
    eprintln!("  Hidden size  : {}", cfg.hidden_size);
    eprintln!("  Layers       : {}", cfg.num_layers);
    eprintln!(
        "  Heads        : {} (KV: {})",
        cfg.num_heads, cfg.num_kv_heads
    );
    eprintln!("  FFN size     : {}", cfg.intermediate_size);
    eprintln!("  Head dim     : {}", cfg.head_dim);
    eprintln!("  Vocab size   : {}", cfg.vocab_size);
    eprintln!("  Max pos      : {}", cfg.max_position_embeddings);
    eprintln!("  RoPE theta   : {}", cfg.rope_theta);
    eprintln!("  RMS norm eps : {}", cfg.rms_norm_eps);
    eprintln!("  Tie embed    : {}", cfg.tie_word_embeddings);
    eprintln!("  Tensors      : {}", bf.tensors.len());
    eprintln!();

    print_tensor_summary(&bf.tensors);
    Ok(())
}

/// Print info from a GGUF file.
fn info_gguf(path: &Path) -> Result<()> {
    let gf = beacon_format::GgufFile::open(path).context("failed to open GGUF file")?;

    let cfg = gf
        .model_config()
        .context("failed to extract model config from GGUF metadata")?;

    eprintln!("{}", "GGUF model file".bold().underline());
    eprintln!("  Path         : {}", path.display());
    eprintln!("  GGUF version : {}", gf.version);
    eprintln!("  Architecture : {:?}", cfg.architecture);
    eprintln!("  Hidden size  : {}", cfg.hidden_size);
    eprintln!("  Layers       : {}", cfg.num_layers);
    eprintln!(
        "  Heads        : {} (KV: {})",
        cfg.num_heads, cfg.num_kv_heads
    );
    eprintln!("  FFN size     : {}", cfg.intermediate_size);
    eprintln!("  Head dim     : {}", cfg.head_dim);
    eprintln!("  Vocab size   : {}", cfg.vocab_size);
    eprintln!("  Metadata KVs : {}", gf.metadata.len());
    eprintln!("  Tensors      : {}", gf.tensors.len());
    eprintln!();

    print_tensor_summary(&gf.tensors);
    Ok(())
}

/// Print a compact summary of tensors grouped by dtype.
fn print_tensor_summary(tensors: &[beacon_format::TensorMeta]) {
    use std::collections::BTreeMap;

    let mut by_dtype: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in tensors {
        let key = format!("{:?}", t.dtype);
        let entry = by_dtype.entry(key).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += t.data_length;
    }

    eprintln!("{}", "Tensor summary".bold().underline());
    for (dtype, (count, bytes)) in &by_dtype {
        #[allow(clippy::cast_precision_loss)]
        let mb = *bytes as f64 / (1024.0 * 1024.0);
        eprintln!("  {dtype:<10} {count:>5} tensors  {mb:>10.1} MB");
    }
}

/// `beacon serve` — placeholder for Step 11.
fn cmd_serve(port: u16) {
    eprintln!(
        "{} {} is not yet implemented (lands in Step 11).",
        "note:".yellow().bold(),
        "beacon serve".bold()
    );
    eprintln!("  Would listen on port {port}.");
    eprintln!("  The HTTP server will provide Ollama-compatible endpoints:");
    eprintln!("    /api/generate, /api/chat, /api/tags, /api/pull");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Default models cache directory: `~/.beacon/models/`.
fn default_models_dir() -> PathBuf {
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
