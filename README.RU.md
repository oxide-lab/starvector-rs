<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

# starvector-rs

Rust crate для локального инференса `starvector-1b-im2svg` на Candle.

## GGUF-веса

Готовые GGUF-файлы модели доступны на Hugging Face:  
https://huggingface.co/oxide-lab/starvector-1b-im2svg-GGUF

## Что уже реализовано

- Reuse-first загрузчик весов из safetensors-шардов.
- Image preprocessor с parity-проверками против локального Python-референса.
- Vision encoder (совместимый layout с Candle CLIP, полный sequence output).
- Adapter с BatchNorm parity.
- Декодер BigCode как тонкая локальная адаптация с KV-cache.
- Путь загрузки GGUF с quantized runtime для decoder/vision/adapter.
- Композиция `StarVector` с greedy или sampling generation (останов по `</svg>`).
- CLI-команда: `starvector-rs infer`.

## Scope (v1)

- Только модель: `starvector-1b-im2svg`.
- Только inference path.
- Greedy и sampling decoding (`temperature`, `top_p`, `repetition_penalty`).

## Сборка

```powershell
cd D:\starvecntor\starvector-rs
cargo build
```

Сборка с CUDA:

```powershell
cd D:\starvecntor\starvector-rs
cargo build --features cuda
```

## Использование CLI

```powershell
cd D:\starvecntor\starvector-rs
cargo run -- infer --model-dir ..\models\starvector-1b-im2svg --image ..\star-vector\assets\examples\sample-18.png --device cpu --max-new-tokens 64
```

Опциональная запись в файл:

```powershell
cargo run -- infer --model-dir ..\models\starvector-1b-im2svg --image ..\star-vector\assets\examples\sample-18.png --device cpu --max-new-tokens 64 --output out.svg
```

Пример с GGUF:

```powershell
cargo run --release --features cuda -- infer --model-dir ..\models\starvector-1b-im2svg --weights-gguf ..\models\starvector-1b-im2svg-full-q4_0.gguf --image ..\star-vector\assets\examples\sample-17.png --device cuda --max-new-tokens 2048 --do-sample true --temperature 0.2 --top-p 0.95 --repetition-penalty 1.1
```

Контракт CLI:

- Печатает итоговый SVG в `stdout`.
- При `--output` дополнительно пишет SVG в файл.
- Ранний stop на `</svg>`.
- `--device cuda` поддерживается и эквивалентен `--device cuda:0`.
- Если бинарь собран **без** feature `cuda`, `--device cuda[:idx]` завершится явной ошибкой.

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

`GenerationOutput.token_ids` содержит только сгенерированные id (без prompt-токенов `<svg`).

## Проверка

Основные проверки:

```powershell
cd D:\starvecntor\starvector-rs
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo build
```

Текущий статус:

- Тесты tokenizer/preprocessor/adapter contracts проходят.
- Greedy generation и CLI CPU path проходят.
- Расширенный Python oracle и отдельный CLI smoke вынесены в следующую задачу.

## Сторонние лицензии

См. [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (сгенерирован через cargo license).

## Лицензия

Apache-2.0

## Благодарности

- Авторам и мейнтейнерам исходной модели `starvector-1b-im2svg`: https://github.com/joanrod/star-vector
- Проекту Candle за Rust ML runtime и поддержку CUDA: https://github.com/huggingface/candle
- Экосистеме инструментов GGUF: https://github.com/ggml-org/llama.cpp и https://github.com/city96/ComfyUI-GGUF

