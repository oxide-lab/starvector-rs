<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

# starvector-rs

Crate Rust para inferência local de `starvector-1b-im2svg` com Candle.

## Pesos GGUF

Arquivos GGUF pré-gerados estão disponíveis no Hugging Face:  
https://huggingface.co/oxide-lab/starvector-1b-im2svg-GGUF

## O que já foi implementado

- Loader de pesos reuse-first com shards safetensors.
- Image preprocessor com testes de paridade contra referência Python local.
- Vision encoder (layout compatível com Candle CLIP, saída de sequência completa).
- Adapter com comportamento parity de BatchNorm.
- Decoder BigCode como adaptação local fina com KV-cache.
- Caminho GGUF com runtime quantizado para decoder/vision/adapter.
- Composição `StarVector` com geração greedy ou sampling (stop em `</svg>`).
- Comando CLI: `starvector-rs infer`.

## Escopo (v1)

- Apenas modelo: `starvector-1b-im2svg`.
- Apenas caminho de inferência.
- Decoding greedy e sampling (`temperature`, `top_p`, `repetition_penalty`).

## Build

```powershell
cd D:\starvecntor\starvector-rs
cargo build
```

Build com CUDA:

```powershell
cd D:\starvecntor\starvector-rs
cargo build --features cuda
```

## Uso da CLI

```powershell
cd D:\starvecntor\starvector-rs
cargo run -- infer --model-dir ..\models\starvector-1b-im2svg --image ..\star-vector\assets\examples\sample-18.png --device cpu --max-new-tokens 64
```

Saída opcional em arquivo:

```powershell
cargo run -- infer --model-dir ..\models\starvector-1b-im2svg --image ..\star-vector\assets\examples\sample-18.png --device cpu --max-new-tokens 64 --output out.svg
```

Exemplo com GGUF:

```powershell
cargo run --release --features cuda -- infer --model-dir ..\models\starvector-1b-im2svg --weights-gguf ..\models\starvector-1b-im2svg-full-q4_0.gguf --image ..\star-vector\assets\examples\sample-17.png --device cuda --max-new-tokens 2048 --do-sample true --temperature 0.2 --top-p 0.95 --repetition-penalty 1.1
```

Contrato da CLI:

- Imprime o SVG resultante em `stdout`.
- Escreve SVG em arquivo quando `--output` é informado.
- Para cedo ao detectar `</svg>`.
- `--device cuda` é suportado e equivale a `--device cuda:0`.
- Em binário compilado **sem** feature `cuda`, `--device cuda[:idx]` falha com erro explícito.

## API de biblioteca

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

`GenerationOutput.token_ids` contém apenas ids gerados (sem os ids do prompt `<svg`).

## Verificação

Checks principais:

```powershell
cd D:\starvecntor\starvector-rs
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo build
```

Status atual:

- Testes de contrato para tokenizer/preprocessor/adapter estão passando.
- Geração greedy e caminho CPU da CLI estão passando.
- Oracle Python estendido e teste dedicado de CLI smoke ficam para a próxima tarefa.

## Licenças de Terceiros

Veja [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (gerado via cargo license).

## Licença

Apache-2.0

## Agradecimentos

- Aos autores e mantenedores do modelo original `starvector-1b-im2svg`: https://github.com/joanrod/star-vector
- Ao projeto Candle pelo runtime de ML em Rust e suporte CUDA: https://github.com/huggingface/candle
- Ao ecossistema de ferramentas GGUF: https://github.com/ggml-org/llama.cpp e https://github.com/city96/ComfyUI-GGUF

