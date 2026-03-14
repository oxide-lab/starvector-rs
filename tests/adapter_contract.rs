use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use candle::{Device, Tensor};
use serde::{Deserialize, Serialize};

use starvector_rs::model::adapter::StarVectorAdapter;
use starvector_rs::model::loader::{LoaderPrecisionPolicy, ReuseFirstWeightLoader};

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

fn require_adapter_parity() -> Result<Option<PathBuf>, Box<dyn Error + Send + Sync>> {
    if !env_is_enabled(RUN_LOCAL_MODEL_TESTS_ENV) {
        eprintln!(
            "adapter_contract: skipped (set {RUN_LOCAL_MODEL_TESTS_ENV}=1 to enable local-model tests)"
        );
        return Ok(None);
    }
    if !env_is_enabled(RUN_PYTHON_ORACLE_ENV) {
        eprintln!(
            "adapter_contract: skipped (set {RUN_PYTHON_ORACLE_ENV}=1 to enable python parity)"
        );
        return Ok(None);
    }
    let dir = model_dir();
    if !dir.exists() {
        eprintln!(
            "adapter_contract: skipped (model dir not found: {})",
            dir.display()
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

fn run_python_preflight(
    script: &std::path::Path,
    model_dir: &std::path::Path,
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
struct AdapterInputPayload {
    shape: [usize; 3],
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct AdapterOracleOutput {
    shape: Vec<usize>,
    values: Vec<f32>,
    num_batches_tracked: i64,
}

#[test]
fn adapter_matches_python_oracle() -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(model_dir) = require_adapter_parity()? else {
        return Ok(());
    };
    run_python_preflight(&python_oracle_script(), &model_dir)?;
    let loader = ReuseFirstWeightLoader::from_model_dir(&model_dir)?;
    let views = loader.make_views(LoaderPrecisionPolicy::cpu_default(), &Device::Cpu)?;
    let adapter = StarVectorAdapter::new(views.adapter, &loader)?;

    let shape = [1, 257, 1024];
    let input_values: Vec<f32> = (0..(shape[0] * shape[1] * shape[2]))
        .map(|i| ((i % 257) as f32) / 257.0 - 0.5)
        .collect();
    let input_tensor = Tensor::from_vec(
        input_values.clone(),
        (shape[0], shape[1], shape[2]),
        &Device::Cpu,
    )?;
    let rust_out = adapter.forward(&input_tensor)?;
    let rust_values = rust_out.flatten_all()?.to_vec1::<f32>()?;

    let input_payload = AdapterInputPayload {
        shape,
        values: input_values,
    };
    let temp_json = std::env::temp_dir().join(format!(
        "starvector_adapter_input_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::write(&temp_json, serde_json::to_vec(&input_payload)?)?;

    let output = Command::new("py")
        .arg("-3")
        .arg(python_oracle_script())
        .arg("adapter")
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--input-json")
        .arg(&temp_json)
        .env("PYTHONPATH", star_vector_dir())
        .output()?;
    let _ = std::fs::remove_file(&temp_json);

    if !output.status.success() {
        return Err(format!(
            "python oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let py_out: AdapterOracleOutput = serde_json::from_slice(&output.stdout)?;
    assert_eq!(py_out.shape, vec![1, 257, 2048]);
    assert_eq!(rust_values.len(), py_out.values.len());
    assert_eq!(adapter.num_batches_tracked(), py_out.num_batches_tracked);

    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    for (a, b) in rust_values.iter().zip(py_out.values.iter()) {
        let diff = (a - b).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        sum_abs_diff += diff as f64;
    }
    let mean_abs_diff = sum_abs_diff / (rust_values.len() as f64);

    assert!(
        max_abs_diff <= 5e-3,
        "adapter parity max_abs_diff too high: {max_abs_diff}"
    );
    assert!(
        mean_abs_diff <= 1e-4,
        "adapter parity mean_abs_diff too high: {mean_abs_diff}"
    );

    Ok(())
}
