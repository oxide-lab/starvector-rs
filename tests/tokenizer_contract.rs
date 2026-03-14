use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use tokenizers::Tokenizer;

use starvector_rs::types::{
    ADDED_TOKENS_FILE, DECODER_EMBEDDING_KEY, MODEL_INDEX_FILE, ParsedModelMetadata,
    TOKENIZER_JSON_FILE,
};

const EXPECTED_VOCAB_SIZE: usize = 49156;
const EXPECTED_BASE_VOCAB_SIZE: usize = 49152;
const EXPECTED_PROMPT_TOKENS: [u32; 2] = [46, 3672];
const EXPECTED_STOP_TOKENS: [u32; 3] = [377, 3672, 48];
const RUN_LOCAL_MODEL_TESTS_ENV: &str = "STARVECTOR_RUN_LOCAL_MODEL_TESTS";
const RUN_PYTHON_ORACLE_ENV: &str = "STARVECTOR_RUN_PYTHON_ORACLE";

fn model_dir() -> PathBuf {
    if let Ok(path) = std::env::var("STARVECTOR_MODEL_DIR") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
        .join("starvector-1b-im2svg")
}

fn env_is_enabled(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

fn require_local_model_tests() -> Result<Option<PathBuf>, Box<dyn Error + Send + Sync>> {
    if !env_is_enabled(RUN_LOCAL_MODEL_TESTS_ENV) {
        eprintln!(
            "tokenizer_contract: skipped (set {RUN_LOCAL_MODEL_TESTS_ENV}=1 to enable local-model tests)"
        );
        return Ok(None);
    }
    let dir = model_dir();
    if !dir.exists() {
        eprintln!(
            "tokenizer_contract: skipped (model dir not found: {})",
            dir.display()
        );
        return Ok(None);
    }
    Ok(Some(dir))
}

fn require_python_oracle_tests() -> Result<Option<PathBuf>, Box<dyn Error + Send + Sync>> {
    let Some(dir) = require_local_model_tests()? else {
        return Ok(None);
    };
    if !env_is_enabled(RUN_PYTHON_ORACLE_ENV) {
        eprintln!(
            "tokenizer_contract: skipped python parity (set {RUN_PYTHON_ORACLE_ENV}=1 to enable)"
        );
        return Ok(None);
    }
    Ok(Some(dir))
}

fn python_oracle_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("python_oracle.py")
}

fn star_vector_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("star-vector")
}

fn run_python_preflight(model_dir: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let output = Command::new("py")
        .arg("-3")
        .arg(python_oracle_script())
        .arg("preflight")
        .arg("--model-dir")
        .arg(model_dir)
        .env("PYTHONPATH", star_vector_dir())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "python preflight failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer, Box<dyn Error + Send + Sync>> {
    let tokenizer_path = model_dir.join(TOKENIZER_JSON_FILE);
    let tokenizer_path = tokenizer_path
        .to_str()
        .ok_or("tokenizer path contains invalid UTF-8")?;
    Ok(Tokenizer::from_file(tokenizer_path)?)
}

fn parse_tensor_shape_from_safetensors_header(
    shard_path: &Path,
    tensor_name: &str,
) -> Result<Vec<usize>, Box<dyn Error + Send + Sync>> {
    let mut file = File::open(shard_path)?;
    let mut header_len_bytes = [0_u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    let mut header_bytes = vec![0_u8; header_len];
    file.read_exact(&mut header_bytes)?;

    let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;
    let shape = header
        .get(tensor_name)
        .and_then(|entry| entry.get("shape"))
        .and_then(|shape| shape.as_array())
        .ok_or_else(|| format!("tensor `{tensor_name}` shape is missing in safetensors header"))?;

    let mut dims = Vec::with_capacity(shape.len());
    for dim in shape {
        let value = dim
            .as_u64()
            .ok_or_else(|| format!("tensor `{tensor_name}` has non-u64 shape dim"))?;
        dims.push(value as usize);
    }
    Ok(dims)
}

#[test]
fn tokenizer_and_model_vocab_contract_is_fixed() -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(model_dir) = require_local_model_tests()? else {
        return Ok(());
    };
    let metadata = ParsedModelMetadata::from_model_dir(&model_dir)?;

    for required in ParsedModelMetadata::required_files() {
        assert!(
            model_dir.join(required).exists(),
            "required metadata file is missing: {required}"
        );
    }

    assert_eq!(metadata.model_config.vocab_size, EXPECTED_VOCAB_SIZE);
    assert_eq!(
        metadata.tokenizer_config.vocab_size,
        EXPECTED_BASE_VOCAB_SIZE
    );
    assert_eq!(metadata.added_tokens.len(), 4);
    assert_eq!(metadata.resolved_vocab_size(), EXPECTED_VOCAB_SIZE);
    assert_eq!(
        metadata.tokenizer_plus_added_vocab_size(),
        EXPECTED_VOCAB_SIZE
    );

    let tokenizer = load_tokenizer(&model_dir)?;
    assert_eq!(
        tokenizer.get_vocab_size(true),
        EXPECTED_VOCAB_SIZE,
        "local tokenizer must already include added tokens"
    );
    assert_eq!(tokenizer.get_vocab_size(false), EXPECTED_BASE_VOCAB_SIZE);

    let index_path = model_dir.join(MODEL_INDEX_FILE);
    let model_index_raw = std::fs::read_to_string(index_path)?;
    let model_index: ModelIndexForTest = serde_json::from_str(&model_index_raw)?;
    let shard_name = model_index
        .weight_map
        .get(DECODER_EMBEDDING_KEY)
        .ok_or("decoder embedding key missing in model.safetensors.index.json")?;

    let embedding_shape = parse_tensor_shape_from_safetensors_header(
        &model_dir.join(shard_name),
        DECODER_EMBEDDING_KEY,
    )?;
    assert_eq!(
        embedding_shape.first().copied(),
        Some(EXPECTED_VOCAB_SIZE),
        "decoder embedding row count must match resolved vocab size"
    );

    Ok(())
}

#[test]
fn special_token_ids_are_frozen() -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(model_dir) = require_local_model_tests()? else {
        return Ok(());
    };
    let metadata = ParsedModelMetadata::from_model_dir(&model_dir)?;
    let token_ids = metadata.special_token_ids();

    assert_eq!(token_ids.pad, 49152);
    assert_eq!(token_ids.svg_start, 49153);
    assert_eq!(token_ids.image_start, 49154);
    assert_eq!(token_ids.caption_start, 49155);

    let tokenizer = load_tokenizer(&model_dir)?;
    assert_eq!(tokenizer.token_to_id("[PAD]"), Some(token_ids.pad));
    assert_eq!(
        tokenizer.token_to_id("<svg-start>"),
        Some(token_ids.svg_start)
    );
    assert_eq!(
        tokenizer.token_to_id("<image-start>"),
        Some(token_ids.image_start)
    );
    assert_eq!(
        tokenizer.token_to_id("<caption-start>"),
        Some(token_ids.caption_start)
    );

    Ok(())
}

#[test]
fn prompt_and_stop_sequences_match_contract() -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(model_dir) = require_local_model_tests()? else {
        return Ok(());
    };
    let tokenizer = load_tokenizer(&model_dir)?;

    let prompt_ids = tokenizer.encode("<svg", false)?.get_ids().to_vec();
    let stop_ids = tokenizer.encode("</svg>", false)?.get_ids().to_vec();

    assert_eq!(prompt_ids, EXPECTED_PROMPT_TOKENS);
    assert_eq!(stop_ids, EXPECTED_STOP_TOKENS);
    Ok(())
}

#[test]
fn encode_decode_parity_with_python_tokenizers_for_fixed_svg()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(model_dir) = require_python_oracle_tests()? else {
        return Ok(());
    };
    run_python_preflight(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;
    let input = "<svg viewBox=\"0 0 32 32\"><path d=\"M0 0h32v32H0z\"/></svg>";

    let rust_encoding = tokenizer.encode(input, false)?;
    let rust_ids = rust_encoding.get_ids().to_vec();
    let rust_decoded = tokenizer.decode(&rust_ids, false)?;

    let python_script = r#"
import json
import sys
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(sys.argv[1])
text = sys.argv[2]
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded.ids, False)
print(json.dumps({"ids": encoded.ids, "decoded": decoded}, ensure_ascii=False))
"#;

    let tokenizer_path = model_dir.join(TOKENIZER_JSON_FILE);
    let output = Command::new("py")
        .arg("-3")
        .arg("-c")
        .arg(python_script)
        .arg(tokenizer_path)
        .arg(input)
        .env("PYTHONPATH", star_vector_dir())
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "python tokenizer oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    #[derive(Debug, Deserialize)]
    struct PythonTokenizerResult {
        ids: Vec<u32>,
        decoded: String,
    }

    let py_result: PythonTokenizerResult = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        rust_ids, py_result.ids,
        "rust/python tokenization ids mismatch"
    );
    assert_eq!(
        rust_decoded, py_result.decoded,
        "rust/python decode result mismatch"
    );

    Ok(())
}

#[test]
fn python_oracle_tokenize_mode_matches_rust_for_fixed_strings()
-> Result<(), Box<dyn Error + Send + Sync>> {
    #[derive(Debug, Deserialize)]
    struct TokenizeOne {
        text: String,
        ids: Vec<u32>,
        decoded: String,
    }
    #[derive(Debug, Deserialize)]
    struct TokenizeResult {
        results: Vec<TokenizeOne>,
    }

    let Some(model_dir) = require_python_oracle_tests()? else {
        return Ok(());
    };
    run_python_preflight(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;
    let tokenizer_path = model_dir.join(TOKENIZER_JSON_FILE);
    let cases = vec![
        "<svg",
        "</svg>",
        "<svg-start>",
        "<svg viewBox=\"0 0 8 8\"><rect width=\"8\" height=\"8\"/></svg>",
    ];

    let mut command = Command::new("py");
    command
        .arg("-3")
        .arg(python_oracle_script())
        .arg("tokenize")
        .arg("--tokenizer-json")
        .arg(tokenizer_path)
        .env("PYTHONPATH", star_vector_dir());
    for case in &cases {
        command.arg("--text").arg(case);
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "python tokenize mode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let parsed: TokenizeResult = serde_json::from_slice(&output.stdout)?;
    assert_eq!(parsed.results.len(), cases.len());
    for item in parsed.results {
        let rust_ids = tokenizer
            .encode(item.text.as_str(), false)?
            .get_ids()
            .to_vec();
        let rust_decoded = tokenizer.decode(&rust_ids, false)?;
        assert_eq!(item.ids, rust_ids, "ids mismatch for {}", item.text);
        assert_eq!(
            item.decoded, rust_decoded,
            "decode mismatch for {}",
            item.text
        );
    }
    Ok(())
}

#[test]
fn local_tokenizer_is_self_sufficient_without_runtime_mutation()
-> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(model_dir) = require_local_model_tests()? else {
        return Ok(());
    };
    let tokenizer = load_tokenizer(&model_dir)?;
    let added_tokens_raw = std::fs::read_to_string(model_dir.join(ADDED_TOKENS_FILE))?;
    let added_tokens: std::collections::HashMap<String, u32> =
        serde_json::from_str(&added_tokens_raw)?;

    for (token, id) in added_tokens {
        assert_eq!(
            tokenizer.token_to_id(&token),
            Some(id),
            "tokenizer.json must already include token `{token}`"
        );
    }

    assert_eq!(
        tokenizer.get_vocab_size(true),
        EXPECTED_VOCAB_SIZE,
        "runtime add_tokens path must not be required"
    );
    Ok(())
}

#[derive(Debug, Deserialize)]
struct ModelIndexForTest {
    weight_map: std::collections::HashMap<String, String>,
}
