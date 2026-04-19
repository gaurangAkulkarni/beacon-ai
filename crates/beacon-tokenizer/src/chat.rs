//! Chat template rendering via `minijinja`.
//!
//! LLM chat models expect prompts formatted according to a model-specific
//! template (stored as a Jinja2 template string in `tokenizer_config.json`
//! or in GGUF metadata). This module renders those templates.

use crate::error::TokenizerError;

/// A chat message with role and content.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render a chat template with the given messages.
///
/// The `template` string is a Jinja2-subset template (as used by
/// `tokenizer_config.json`). Common variables available in the template:
///
/// - `messages`: array of `{role, content}` objects
/// - `bos_token`, `eos_token`: special token strings
/// - `add_generation_prompt`: whether to add the assistant prompt prefix
pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    add_generation_prompt: bool,
) -> Result<String, TokenizerError> {
    let mut env = minijinja::Environment::new();
    env.add_template("chat", template)
        .map_err(|e| TokenizerError::Template(format!("invalid template: {e}")))?;

    let tmpl = env
        .get_template("chat")
        .map_err(|e| TokenizerError::Template(format!("template lookup: {e}")))?;

    let ctx = minijinja::context! {
        messages => messages,
        bos_token => bos_token.unwrap_or(""),
        eos_token => eos_token.unwrap_or(""),
        add_generation_prompt => add_generation_prompt,
    };

    tmpl.render(ctx)
        .map_err(|e| TokenizerError::Template(format!("render error: {e}")))
}
