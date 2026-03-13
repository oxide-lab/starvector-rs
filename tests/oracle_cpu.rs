use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle::{Device, Tensor};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use starvector_rs::model::adapter::StarVectorAdapter;
use starvector_rs::model::bigcode::{BigCodeConfig, BigCodeDecoder};
use starvector_rs::model::generation::{greedy_next_token_id, has_suffix};
use starvector_rs::model::image_preprocessor::ImagePreprocessor;
use starvector_rs::model::loader::{LoaderPrecisionPolicy, ReuseFirstWeightLoader};
use starvector_rs::model::vision::{StarVectorVisionConfig, StarVectorVisionTransformer};
use starvector_rs::types::{ParsedModelMetadata, TOKENIZER_JSON_FILE};

fn model_dir() -> PathBuf {
    if let Ok(path) = std::env::var("STARVECTOR_MODEL_DIR") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
        .join("starvector-1b-im2svg")
}

fn sample_image() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("star-vector")
        .join("assets")
        .join("examples")
        .join("sample-18.png")
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

fn run_python_preflight(
    script: &Path,
    model_dir: &Path,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let output = Command::new("py")
        .arg("-3")
        .arg(script)
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

#[derive(Debug, Serialize)]
struct PrefixEmbedsInput {
    shape: [usize; 3],
    values: Vec<f32>,
    stop_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct GreedyTraceOutput {
    token_ids: Vec<u32>,
}

#[test]
fn oracle_cpu_greedy_step_trace_matches_rust_decoder() -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_dir = model_dir();
    let image = sample_image();
    let script = python_oracle_script();
    run_python_preflight(&script, &model_dir)?;

    let parsed = ParsedModelMetadata::from_model_dir(&model_dir)?;
    let loader = ReuseFirstWeightLoader::from_model_dir(&model_dir)?;
    let views = loader.make_views(LoaderPrecisionPolicy::cpu_default(), &Device::Cpu)?;

    let preprocessor = ImagePreprocessor::from_model_dir(&model_dir)?;
    let mut vision_cfg = StarVectorVisionConfig {
        image_size: parsed.model_config.image_size,
        ..Default::default()
    };
    if let Some(scale) = parsed.model_config.hidden_size_scale
        && scale > 0
    {
        vision_cfg.embed_dim = parsed.model_config.hidden_size / scale;
        vision_cfg.intermediate_size = vision_cfg.embed_dim * 4;
    }
    let vision = StarVectorVisionTransformer::new(views.vision, &vision_cfg)?;
    let adapter = StarVectorAdapter::new(views.adapter, &loader)?;
    let mut decoder = BigCodeDecoder::load(
        views.decoder,
        BigCodeConfig::from_model_config(&parsed.model_config),
    )?;

    let tokenizer_path = model_dir.join(TOKENIZER_JSON_FILE);
    let tokenizer_path = tokenizer_path
        .to_str()
        .ok_or("tokenizer path contains invalid UTF-8")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let prompt_ids = tokenizer.encode("<svg", false)?.get_ids().to_vec();
    let stop_ids = tokenizer.encode("</svg>", false)?.get_ids().to_vec();

    let pixel = preprocessor
        .preprocess_to_tensor(&image, &Device::Cpu)?
        .unsqueeze(0)?;
    let visual = vision.forward_sequence(&pixel)?;
    let visual = adapter.forward(&visual)?;

    let prompt = Tensor::from_slice(&prompt_ids, (1, prompt_ids.len()), &Device::Cpu)?;
    let prompt_embeds = decoder.embed_tokens(&prompt)?;
    let prefix_embeds = Tensor::cat(&[&visual, &prompt_embeds], 1)?;

    let (b, seq, hidden) = prefix_embeds.dims3()?;
    assert_eq!(b, 1);
    let prefix_values = prefix_embeds.flatten_all()?.to_vec1::<f32>()?;

    let payload = PrefixEmbedsInput {
        shape: [b, seq, hidden],
        values: prefix_values,
        stop_ids: stop_ids.clone(),
    };
    let temp_json = std::env::temp_dir().join(format!(
        "starvector_prefix_embeds_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::write(&temp_json, serde_json::to_vec(&payload)?)?;

    let max_new_tokens = 12usize;
    let py = Command::new("py")
        .arg("-3")
        .arg(&script)
        .arg("greedy_step_trace")
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--input-json")
        .arg(&temp_json)
        .arg("--max-new-tokens")
        .arg(max_new_tokens.to_string())
        .arg("--device")
        .arg("cpu")
        .arg("--tokenizer-json")
        .arg(model_dir.join(TOKENIZER_JSON_FILE))
        .env("PYTHONPATH", star_vector_dir())
        .output()?;
    let _ = std::fs::remove_file(&temp_json);
    if !py.status.success() {
        return Err(format!(
            "python greedy_step_trace failed: {}",
            String::from_utf8_lossy(&py.stderr)
        )
        .into());
    }
    let py_out: GreedyTraceOutput = serde_json::from_slice(&py.stdout)?;

    decoder.clear_kv_cache();
    let mut logits = decoder.forward_inputs_embeds(&prefix_embeds, 0)?;
    let mut rust_ids = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let next = greedy_next_token_id(&logits)?;
        rust_ids.push(next);
        if has_suffix(&rust_ids, &stop_ids) {
            break;
        }
        let input = Tensor::from_slice(&[next], (1, 1), &Device::Cpu)?;
        let past_len = seq + rust_ids.len() - 1;
        logits = decoder.forward(&input, past_len)?;
    }

    assert_eq!(
        rust_ids, py_out.token_ids,
        "rust/python greedy token trace mismatch"
    );
    Ok(())
}
