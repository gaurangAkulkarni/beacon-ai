//! `BeaconTokenizer` — safe wrapper around the `HuggingFace` `tokenizers` crate.
//!
//! Provides encode/decode plus chat template rendering. Loads from
//! `tokenizer.json` files (the standard format shipped with every HF model).

use std::path::Path;

use crate::chat::{self, ChatMessage};
use crate::error::TokenizerError;

/// A tokenizer wrapping the `HuggingFace` `tokenizers` crate with chat template
/// support.
///
/// Create via [`BeaconTokenizer::from_file`] (loading a `tokenizer.json`) or
/// [`BeaconTokenizer::from_bytes`] (loading from in-memory JSON bytes).
pub struct BeaconTokenizer {
    inner: tokenizers::Tokenizer,
    /// Optional Jinja2 chat template string.
    chat_template: Option<String>,
    /// BOS token string (resolved from the tokenizer's added tokens).
    bos_token: Option<String>,
    /// EOS token string (resolved from the tokenizer's added tokens).
    eos_token: Option<String>,
}

impl std::fmt::Debug for BeaconTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeaconTokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("has_chat_template", &self.chat_template.is_some())
            .finish_non_exhaustive()
    }
}

impl BeaconTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self {
            inner,
            chat_template: None,
            bos_token: None,
            eos_token: None,
        })
    }

    /// Load a tokenizer from in-memory JSON bytes (the content of a
    /// `tokenizer.json`).
    pub fn from_bytes(json: &[u8]) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_bytes(json)
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self {
            inner,
            chat_template: None,
            bos_token: None,
            eos_token: None,
        })
    }

    /// Set the chat template (Jinja2 subset).
    #[must_use]
    pub fn with_chat_template(mut self, template: impl Into<String>) -> Self {
        self.chat_template = Some(template.into());
        self
    }

    /// Set special token strings used in chat template rendering.
    #[must_use]
    pub fn with_special_tokens(
        mut self,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        self.bos_token = bos_token;
        self.eos_token = eos_token;
        self
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode text into token IDs.
    ///
    /// If `add_special_tokens` is true, BOS/EOS and other special tokens
    /// defined by the tokenizer are added automatically.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    ///
    /// If `skip_special_tokens` is true, special tokens (BOS, EOS, PAD, etc.)
    /// are omitted from the output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Look up the token ID for a string. Returns `None` if not in vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Look up the string for a token ID. Returns `None` if out of range.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Apply the chat template to a list of messages, returning the formatted
    /// prompt string.
    ///
    /// `add_generation_prompt` controls whether the template appends the
    /// assistant turn prefix (set to `true` for inference, `false` for
    /// training).
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let template = self
            .chat_template
            .as_deref()
            .ok_or(TokenizerError::NoChatTemplate)?;

        chat::render_chat_template(
            template,
            messages,
            self.bos_token.as_deref(),
            self.eos_token.as_deref(),
            add_generation_prompt,
        )
    }

    /// Whether a chat template is configured.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Access the underlying `tokenizers::Tokenizer` (escape hatch).
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
