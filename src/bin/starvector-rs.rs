use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};
use starvector_rs::{GenerationConfig, PrecisionPolicy, RuntimeDevice, StarVector};

#[derive(Debug, Parser)]
#[command(
    name = "starvector-rs",
    version,
    about = "StarVector CLI for local im2svg inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run im2svg inference and print SVG to stdout.
    Infer {
        /// Path to local model directory (for example ..\\models\\starvector-1b-im2svg).
        #[arg(long)]
        model_dir: PathBuf,
        /// Input raster image path.
        #[arg(long)]
        image: PathBuf,
        /// Device selector: cpu | cuda | cuda:<index>.
        #[arg(long, default_value = "cpu")]
        device: RuntimeDevice,
        /// Optional GGUF weights path. When set, tensors are loaded from GGUF instead of safetensors shards.
        #[arg(long)]
        weights_gguf: Option<PathBuf>,
        /// Maximum number of generated tokens after the <svg prompt.
        #[arg(long, default_value_t = 7933)]
        max_new_tokens: usize,
        /// Enable sampling instead of greedy decoding.
        #[arg(long, default_value_t = false, action = ArgAction::Set)]
        do_sample: bool,
        /// Force greedy decoding (overrides --do-sample).
        #[arg(long, default_value_t = false)]
        greedy: bool,
        /// Sampling temperature (used when --do-sample=true).
        #[arg(long, default_value_t = 0.2)]
        temperature: f64,
        /// Nucleus sampling top-p (used when --do-sample=true).
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
        /// Repetition penalty (>1.0 penalizes repeated tokens).
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f64,
        /// RNG seed for sampling.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional path to also write SVG output into a file.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Infer {
            model_dir,
            image,
            device,
            weights_gguf,
            max_new_tokens,
            do_sample,
            greedy,
            temperature,
            top_p,
            repetition_penalty,
            seed,
            output,
        } => {
            let candle_device = device.to_candle_device()?;
            let precision = PrecisionPolicy::for_device(&candle_device);
            let mut model =
                StarVector::load_with_gguf(model_dir, weights_gguf, &candle_device, precision)?;
            let output_svg = model.generate_svg(
                image,
                &GenerationConfig {
                    max_new_tokens,
                    do_sample: if greedy { false } else { do_sample },
                    temperature,
                    top_p,
                    repetition_penalty,
                    seed,
                },
            )?;

            println!("{}", output_svg.svg);
            if let Some(path) = output {
                std::fs::write(path, output_svg.svg)?;
            }
        }
    }
    Ok(())
}
