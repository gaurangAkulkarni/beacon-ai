//! Tests for `beacon-tokenizer`.
//!
//! Uses a programmatically-built BPE tokenizer for unit tests (no external
//! files). A real-model test is gated behind the `BEACON_TEST_TOKENIZER`
//! environment variable.

use crate::{BeaconTokenizer, ChatMessage};

/// Build a minimal tokenizer JSON string for testing.
///
/// Uses the `tokenizers` crate's builder API to create a tiny BPE tokenizer,
/// serializes it to JSON, and returns the bytes.
/// Build a minimal valid `tokenizer.json` for testing.
///
/// Constructs a `WordLevel` tokenizer (simplest model type) via JSON, which
/// the HF `tokenizers` crate can load. This avoids fighting with the BPE
/// builder's `AHashMap` requirements.
fn build_test_tokenizer_json() -> Vec<u8> {
    let json = r#"{
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": {
                "hello": 0,
                "world": 1,
                ",": 2,
                " ": 3,
                "!": 4,
                "the": 5,
                "a": 6,
                "<unk>": 7
            },
            "unk_token": "<unk>"
        },
        "pre_tokenizer": {
            "type": "Whitespace"
        }
    }"#;
    json.as_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// Tokenizer round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn load_from_bytes() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json).unwrap();
    assert!(tok.vocab_size() > 0);
}

#[test]
fn encode_decode_round_trip() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json).unwrap();

    let text = "hello";
    let ids = tok.encode(text, false).unwrap();
    assert!(!ids.is_empty(), "encoding should produce tokens");

    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, text, "decode(encode(x)) should round-trip");
}

#[test]
fn token_id_lookup() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json).unwrap();

    // Check that we can look up a known token.
    if let Some(id) = tok.token_to_id("hello") {
        let back = tok.id_to_token(id);
        assert_eq!(back.as_deref(), Some("hello"));
    }
}

// ---------------------------------------------------------------------------
// Chat template tests
// ---------------------------------------------------------------------------

#[test]
fn chat_template_basic() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json).unwrap().with_chat_template(
        // Simple ChatML-style template.
        "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    );

    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: "Hello!".to_owned(),
    }];

    let rendered = tok.apply_chat_template(&messages, true).unwrap();
    assert!(rendered.contains("Hello!"));
    assert!(rendered.contains("<|im_start|>user"));
    assert!(rendered.contains("<|im_start|>assistant"));
}

#[test]
fn chat_template_no_generation_prompt() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json).unwrap().with_chat_template(
        "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    );

    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: "Hello!".to_owned(),
    }];

    let rendered = tok.apply_chat_template(&messages, false).unwrap();
    assert!(rendered.contains("Hello!"));
    assert!(
        !rendered.contains("<|im_start|>assistant"),
        "should not contain assistant prompt when add_generation_prompt=false"
    );
}

#[test]
fn chat_template_multi_turn() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json)
        .unwrap()
        .with_chat_template(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}",
        );

    let messages = vec![
        ChatMessage {
            role: "system".to_owned(),
            content: "You are helpful.".to_owned(),
        },
        ChatMessage {
            role: "user".to_owned(),
            content: "Hi".to_owned(),
        },
        ChatMessage {
            role: "assistant".to_owned(),
            content: "Hello!".to_owned(),
        },
    ];

    let rendered = tok.apply_chat_template(&messages, false).unwrap();
    assert!(rendered.contains("system: You are helpful."));
    assert!(rendered.contains("user: Hi"));
    assert!(rendered.contains("assistant: Hello!"));
}

#[test]
fn no_chat_template_returns_error() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json).unwrap();

    let result = tok.apply_chat_template(&[], false);
    assert!(result.is_err());
}

#[test]
fn with_special_tokens() {
    let json = build_test_tokenizer_json();
    let tok = BeaconTokenizer::from_bytes(&json)
        .unwrap()
        .with_chat_template(
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}",
        )
        .with_special_tokens(Some("<s>".to_owned()), Some("</s>".to_owned()));

    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: "test".to_owned(),
    }];

    let rendered = tok.apply_chat_template(&messages, false).unwrap();
    assert_eq!(rendered, "<s>test</s>");
}

// ---------------------------------------------------------------------------
// Real tokenizer test (ignored by default)
// ---------------------------------------------------------------------------

/// Load a real `tokenizer.json` from disk and verify encode/decode.
///
/// Run with: `BEACON_TEST_TOKENIZER=/path/to/tokenizer.json cargo test -p beacon-tokenizer -- --ignored`
#[test]
#[ignore = "requires BEACON_TEST_TOKENIZER env var pointing to a tokenizer.json"]
fn real_tokenizer_encode_decode() {
    let path = std::env::var("BEACON_TEST_TOKENIZER")
        .expect("set BEACON_TEST_TOKENIZER=/path/to/tokenizer.json");

    let tok = BeaconTokenizer::from_file(std::path::Path::new(&path)).unwrap();

    eprintln!("Vocab size: {}", tok.vocab_size());

    let test_strings = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "こんにちは世界",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        " leading space",
        "",
    ];

    for text in &test_strings {
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids, false).unwrap();
        eprintln!("  {text:?} → {ids:?} → {decoded:?}");
        assert_eq!(
            &decoded, text,
            "encode/decode round-trip failed for {text:?}"
        );
    }
}
