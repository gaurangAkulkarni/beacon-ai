//! Model name resolution: short aliases and `repo:filename` parsing.

use crate::error::RegistryError;

/// Describes where to fetch a model's GGUF file and its corresponding
/// tokenizer from `HuggingFace` Hub.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// `HuggingFace` repo ID for the GGUF file (e.g.
    /// `"Qwen/Qwen2.5-0.5B-Instruct-GGUF"`).
    pub repo: String,
    /// Filename of the GGUF file within the repo (e.g.
    /// `"qwen2.5-0.5b-instruct-q4_k_m.gguf"`).
    pub gguf_file: String,
    /// `HuggingFace` repo ID that contains `tokenizer.json` (e.g.
    /// `"Qwen/Qwen2.5-0.5B-Instruct"`).
    pub tokenizer_repo: String,
    /// Short display name used for the local cache directory.
    pub display_name: String,
}

/// Built-in model alias table. Maps user-friendly short names to full
/// `HuggingFace` Hub coordinates. Hardcoded for v0.2; a config-file-based
/// registry is a future enhancement.
const ALIASES: &[(&str, &str, &str, &str)] = &[
    // (alias, gguf_repo, gguf_filename, tokenizer_repo)
    (
        "qwen2.5-0.5b",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ),
    (
        "qwen2.5-1.5b",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ),
    (
        "qwen2.5-3b",
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-3B-Instruct",
    ),
    (
        "qwen2.5-7b",
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
];

/// Resolve a user-provided model name to a [`ModelSpec`].
///
/// Supports three input forms:
///
/// 1. **Short alias** — e.g. `"qwen2.5-0.5b"`, looked up in the built-in
///    alias table.
/// 2. **`repo:filename`** — e.g.
///    `"Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-fp16.gguf"`.
///    The tokenizer repo is inferred by stripping the `-GGUF` suffix from the
///    repo name.
/// 3. **Full repo path** (no colon, contains `/`) — uses the first `.gguf`
///    file convention. This form is not fully supported yet and returns an
///    error suggesting the `repo:filename` syntax.
pub fn resolve_model(name: &str) -> Result<ModelSpec, RegistryError> {
    // 1. Check built-in aliases (case-insensitive).
    let lower = name.to_lowercase();
    for &(alias, repo, gguf_file, tokenizer_repo) in ALIASES {
        if lower == alias {
            return Ok(ModelSpec {
                repo: repo.to_owned(),
                gguf_file: gguf_file.to_owned(),
                tokenizer_repo: tokenizer_repo.to_owned(),
                display_name: alias.to_owned(),
            });
        }
    }

    // 2. `repo:filename` syntax.
    if let Some((repo, filename)) = name.split_once(':') {
        if repo.contains('/') {
            let tokenizer_repo = repo.strip_suffix("-GGUF").unwrap_or(repo);
            let display = filename
                .strip_suffix(".gguf")
                .unwrap_or(filename)
                .to_owned();
            return Ok(ModelSpec {
                repo: repo.to_owned(),
                gguf_file: filename.to_owned(),
                tokenizer_repo: tokenizer_repo.to_owned(),
                display_name: display,
            });
        }
    }

    // 3. Bare repo path without filename — we cannot infer the GGUF filename.
    if name.contains('/') {
        return Err(RegistryError::UnknownModel(format!(
            "{name}\n\
             Hint: use repo:filename syntax, e.g.:\n  \
             beacon pull {name}:<filename>.gguf"
        )));
    }

    // Unknown alias.
    let available: Vec<&str> = ALIASES.iter().map(|(a, _, _, _)| *a).collect();
    Err(RegistryError::UnknownModel(format!(
        "{name}\n\
         Available models: {}\n\
         Or use repo:filename syntax, e.g.:\n  \
         beacon pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf",
        available.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_alias() {
        let spec = resolve_model("qwen2.5-0.5b").unwrap();
        assert_eq!(spec.repo, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(spec.gguf_file, "qwen2.5-0.5b-instruct-q4_k_m.gguf");
        assert_eq!(spec.tokenizer_repo, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(spec.display_name, "qwen2.5-0.5b");
    }

    #[test]
    fn resolve_repo_colon_filename() {
        let spec = resolve_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF:some-file.gguf").unwrap();
        assert_eq!(spec.repo, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(spec.gguf_file, "some-file.gguf");
        assert_eq!(spec.tokenizer_repo, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(spec.display_name, "some-file");
    }

    #[test]
    fn resolve_unknown_returns_error() {
        let err = resolve_model("nonexistent-model").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nonexistent-model"));
        assert!(msg.contains("qwen2.5-0.5b"));
    }

    #[test]
    fn resolve_bare_repo_returns_error() {
        let err = resolve_model("Qwen/SomeRepo").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("repo:filename"));
    }
}
