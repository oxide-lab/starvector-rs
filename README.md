<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

# starvector-rs

Rust crate for local `starvector-1b-im2svg` inference on Candle.

## GGUF Weights

Prebuilt GGUF model files are available on Hugging Face:  
https://huggingface.co/oxide-lab/starvector-1b-im2svg-GGUF

## What Is Implemented

- Reuse-first weight loader over safetensors shards.
- Image preprocessor with parity checks against local Python reference.
- Vision encoder (Candle CLIP-compatible layout, full sequence output).
- Adapter with BatchNorm parity behavior.
- BigCode decoder as a thin local adaptation with KV-cache support.
- GGUF loading path with quantized runtime for decoder/vision/adapter.
- `StarVector` composition with greedy or sampling generation (`</svg>` stop sequence).
- CLI command: `starvector-rs infer`.

## Scope (v1)

- Model: `starvector-1b-im2svg` only.
- Path: inference only.
- Decoding: greedy and sampling (`temperature`, `top_p`, `repetition_penalty`).

## Build

```powershell
cd D:\starvecntor\starvector-rs
cargo build
```

CUDA build:

```powershell
cd D:\starvecntor\starvector-rs
cargo build --features cuda
```

## CLI Usage

```powershell
cd D:\starvecntor\starvector-rs
cargo run -- infer --model-dir ..\models\starvector-1b-im2svg --image ..\star-vector\assets\examples\sample-18.png --device cpu --max-new-tokens 64
```

Optional file output:

```powershell
cargo run -- infer --model-dir ..\models\starvector-1b-im2svg --image ..\star-vector\assets\examples\sample-18.png --device cpu --max-new-tokens 64 --output out.svg
```

GGUF example:

```powershell
cargo run --release --features cuda -- infer --model-dir ..\models\starvector-1b-im2svg --weights-gguf ..\models\starvector-1b-im2svg-full-q4_0.gguf --image ..\star-vector\assets\examples\sample-17.png --device cuda --max-new-tokens 2048 --do-sample true --temperature 0.2 --top-p 0.95 --repetition-penalty 1.1
```

CLI contract:

- Prints resulting SVG to `stdout`.
- Writes SVG to file when `--output` is provided.
- Stops early on `</svg>`.
- `--device cuda` is supported and is equivalent to `--device cuda:0`.
- On binaries built **without** `cuda` feature, `--device cuda[:idx]` fails with an explicit error.

## Library API

```rust
use candle::Device;
use starvector_rs::{GenerationConfig, PrecisionPolicy, StarVector};

let device = Device::Cpu;
let mut model = StarVector::load(
    "../models/starvector-1b-im2svg",
    &device,
    PrecisionPolicy::for_device(&device),
)?;

let out = model.generate_svg("../star-vector/assets/examples/sample-18.png", &GenerationConfig {
    max_new_tokens: 64,
})?;

println!("{}", out.svg);
```

`GenerationOutput.token_ids` contains only generated ids (without the `<svg` prompt ids).

## Verification

Primary checks:

```powershell
cd D:\starvecntor\starvector-rs
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo build
```

Current status:

- Rust unit/integration tests for tokenizer/preprocessor/adapter contracts are passing.
- Greedy generation and CLI CPU path are passing.
- Extended Python oracle and dedicated CLI smoke tests are tracked in the next task.

## Third-Party Licenses

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (generated via cargo license).

## License

Apache-2.0

## Acknowledgements

- The original `starvector-1b-im2svg` model authors and maintainers: https://github.com/joanrod/star-vector
- The Candle project for Rust ML runtime and CUDA support: https://github.com/huggingface/candle
- The GGUF tooling ecosystem: https://github.com/ggml-org/llama.cpp and https://github.com/city96/ComfyUI-GGUF

