use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use starvector_rs::model::image_preprocessor::ImagePreprocessor;

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

fn require_preprocessor_parity() -> Result<Option<PathBuf>, Box<dyn Error + Send + Sync>> {
    if !env_is_enabled(RUN_LOCAL_MODEL_TESTS_ENV) {
        eprintln!(
            "preprocessor_contract: skipped (set {RUN_LOCAL_MODEL_TESTS_ENV}=1 to enable local-model tests)"
        );
        return Ok(None);
    }
    if !env_is_enabled(RUN_PYTHON_ORACLE_ENV) {
        eprintln!(
            "preprocessor_contract: skipped (set {RUN_PYTHON_ORACLE_ENV}=1 to enable python parity)"
        );
        return Ok(None);
    }
    let dir = model_dir();
    if !dir.exists() {
        eprintln!(
            "preprocessor_contract: skipped (model dir not found: {})",
            dir.display()
        );
        return Ok(None);
    }
    Ok(Some(dir))
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

#[derive(Debug, Deserialize)]
struct PythonPreprocessOutput {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn run_python_preprocess(
    script: &Path,
    model_dir: &Path,
    image: &Path,
) -> Result<PythonPreprocessOutput, Box<dyn Error + Send + Sync>> {
    let output = Command::new("py")
        .arg("-3")
        .arg(script)
        .arg("preprocess")
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--image")
        .arg(image)
        .env("PYTHONPATH", star_vector_dir())
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "python oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let payload: PythonPreprocessOutput = serde_json::from_slice(&output.stdout)?;
    Ok(payload)
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

#[test]
fn rust_preprocessor_matches_python_reference() -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(model_dir) = require_preprocessor_parity()? else {
        return Ok(());
    };
    let image_path = sample_image();
    let script = python_oracle_script();
    run_python_preflight(&script, &model_dir)?;

    let preprocessor = ImagePreprocessor::from_model_dir(&model_dir)?;
    let rust = preprocessor.preprocess_to_chw_vec(&image_path)?;
    let py = run_python_preprocess(&script, &model_dir, &image_path)?;

    assert_eq!(py.shape, vec![3, preprocessor.size(), preprocessor.size()]);
    assert_eq!(rust.len(), py.values.len());

    let mut max_abs_diff = 0.0_f32;
    let mut sum_abs_diff = 0.0_f64;
    for (a, b) in rust.iter().zip(py.values.iter()) {
        let diff = (a - b).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        sum_abs_diff += diff as f64;
    }
    let mean_abs_diff = sum_abs_diff / (rust.len() as f64);

    // Bicubic interpolation kernels differ slightly between PIL and image crate.
    assert!(
        max_abs_diff <= 0.08,
        "max_abs_diff too high: {max_abs_diff}"
    );
    assert!(
        mean_abs_diff <= 0.005,
        "mean_abs_diff too high: {mean_abs_diff}"
    );

    Ok(())
}
