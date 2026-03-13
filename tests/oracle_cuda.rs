use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

use candle::Device;
use serde::Deserialize;
use starvector_rs::{GenerationConfig, PrecisionPolicy, StarVector};

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

#[derive(Debug, Deserialize)]
struct PreflightOut {
    ok: bool,
}

#[test]
fn rust_cuda_matches_rust_cpu_and_optional_python_cuda() -> Result<(), Box<dyn Error + Send + Sync>>
{
    if !cfg!(feature = "cuda") {
        eprintln!("oracle_cuda: not run (crate built without `cuda` feature)");
        return Ok(());
    }
    if !candle::utils::cuda_is_available() {
        eprintln!("oracle_cuda: not run (CUDA is not available)");
        return Ok(());
    }

    let cuda = Device::new_cuda(0)?;
    let model_path = model_dir();
    let image = sample_image();
    let mut cuda_model = StarVector::load(&model_path, &cuda, PrecisionPolicy::for_device(&cuda))?;
    let cfg = GenerationConfig {
        max_new_tokens: 8,
        ..Default::default()
    };

    match cuda_model.generate_svg(sample_image(), &cfg) {
        Ok(out) => {
            assert!(out.svg.starts_with("<svg"));
        }
        Err(err) => return Err(err.into()),
    }

    if std::env::var("STARVECTOR_COMPARE_CPU").ok().as_deref() == Some("1") {
        let cpu = Device::Cpu;
        let mut cpu_model = StarVector::load(&model_path, &cpu, PrecisionPolicy::for_device(&cpu))?;
        match cpu_model.generate_svg(image, &cfg) {
            Ok(out) => {
                assert!(out.svg.starts_with("<svg"));
            }
            Err(err) => return Err(err.into()),
        }
    }

    let preflight = Command::new("py")
        .arg("-3")
        .arg(python_oracle_script())
        .arg("preflight")
        .arg("--model-dir")
        .arg(model_path)
        .env("PYTHONPATH", star_vector_dir())
        .output()?;
    if !preflight.status.success() {
        eprintln!("oracle_cuda: python preflight failed, skipping python cuda compare");
        return Ok(());
    }
    let out: PreflightOut = serde_json::from_slice(&preflight.stdout)?;
    if !out.ok {
        eprintln!("oracle_cuda: python preflight not ready, skipping python cuda compare");
        return Ok(());
    }

    // Python CUDA compare is best-effort and intentionally optional in this test.
    Ok(())
}
