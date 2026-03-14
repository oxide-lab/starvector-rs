use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

const RUN_LOCAL_MODEL_TESTS_ENV: &str = "STARVECTOR_RUN_LOCAL_MODEL_TESTS";
const RUN_SAFE_1B_CUDA_ENV: &str = "STARVECTOR_RUN_SAFE_1B_CUDA";

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

fn sample_image() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("star-vector")
        .join("assets")
        .join("examples")
        .join("sample-18.png")
}

#[test]
fn infer_cli_outputs_svg_and_writes_file() -> Result<(), Box<dyn Error + Send + Sync>> {
    if !env_is_enabled(RUN_LOCAL_MODEL_TESTS_ENV) {
        eprintln!(
            "cli_smoke: skipped (set {RUN_LOCAL_MODEL_TESTS_ENV}=1 to enable local-model tests)"
        );
        return Ok(());
    }
    if !env_is_enabled(RUN_SAFE_1B_CUDA_ENV) {
        eprintln!("cli_smoke: skipped (set {RUN_SAFE_1B_CUDA_ENV}=1 to enable CUDA smoke)");
        return Ok(());
    }
    if !cfg!(feature = "cuda") {
        eprintln!("cli_smoke: not run (crate built without `cuda` feature)");
        return Ok(());
    }
    if !candle::utils::cuda_is_available() {
        eprintln!("cli_smoke: not run (CUDA is not available)");
        return Ok(());
    }
    if !model_dir().exists() {
        eprintln!(
            "cli_smoke: skipped (model dir not found: {})",
            model_dir().display()
        );
        return Ok(());
    }

    let out_path = std::env::temp_dir().join(format!(
        "starvector_cli_smoke_{}_{}.svg",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));

    let output = Command::new(env!("CARGO_BIN_EXE_starvector-rs"))
        .arg("infer")
        .arg("--model-dir")
        .arg(model_dir())
        .arg("--image")
        .arg(sample_image())
        .arg("--device")
        .arg("cuda:0")
        .arg("--max-new-tokens")
        .arg("256")
        .arg("--output")
        .arg(&out_path)
        .output()?;

    assert!(
        output.status.success(),
        "cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(!stdout.trim().is_empty(), "stdout is empty");
    assert!(stdout.trim_start().starts_with("<svg"));

    assert!(out_path.exists(), "output file was not created");
    let file_svg = std::fs::read_to_string(&out_path)?;
    assert!(file_svg.trim_start().starts_with("<svg"));
    let _ = std::fs::remove_file(out_path);
    Ok(())
}
