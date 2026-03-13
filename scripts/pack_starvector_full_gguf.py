from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


def _add_llama_cpp_gguf_to_path(repo_root: Path) -> None:
    gguf_py = repo_root / "references" / "llama.cpp" / "gguf-py"
    if not gguf_py.exists():
        raise FileNotFoundError(f"gguf-py not found at {gguf_py}")
    sys.path.insert(0, str(gguf_py))


def _numpy_from_reader(reader, key: str) -> np.ndarray:
    t = reader.get_tensor(key)
    return t.cpu().numpy()

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack full StarVector model (vision + adapter + decoder) into a single unquantized GGUF container."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        type=str,
        default="f16",
        choices=["f16", "q8_0", "q4_0"],
        help="Tensor quantization mode for GGUF weights.",
    )
    parser.add_argument(
        "--strict-quantization",
        action="store_true",
        help="Fail if any tensor cannot be quantized to the selected mode.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    model_dir = args.model_dir.resolve()
    out_path = args.out.resolve()
    quant_mode = args.quantization
    strict_quant = args.strict_quantization

    _add_llama_cpp_gguf_to_path(repo_root)
    import gguf  # type: ignore
    from gguf.quants import quantize as gguf_quantize  # type: ignore

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    writer = gguf.GGUFWriter(str(out_path), "starvector")
    writer.add_name("starvector-1b-im2svg-full")
    writer.add_description("Full StarVector package: image encoder + adapter + decoder")
    writer.add_key_value("starvector.model_type", "starvector", gguf.GGUFValueType.STRING)
    writer.add_key_value(
        "starvector.config_json",
        json.dumps(config, separators=(",", ":"), ensure_ascii=False),
        gguf.GGUFValueType.STRING,
    )
    writer.add_key_value(
        "starvector.source_model_dir",
        str(model_dir),
        gguf.GGUFValueType.STRING,
    )
    writer.add_key_value(
        "starvector.quantization",
        quant_mode,
        gguf.GGUFValueType.STRING,
    )
    writer.add_key_value(
        "starvector.strict_quantization",
        strict_quant,
        gguf.GGUFValueType.BOOL,
    )

    file_names = sorted([p.name for p in model_dir.iterdir() if p.is_file()])
    writer.add_key_value(
        "starvector.files",
        file_names,
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.STRING,
    )

    shards = [
        model_dir / "model-00001-of-00002.safetensors",
        model_dir / "model-00002-of-00002.safetensors",
    ]

    quant_type = None
    if quant_mode == "q8_0":
        quant_type = gguf.GGMLQuantizationType.Q8_0
    elif quant_mode == "q4_0":
        quant_type = gguf.GGMLQuantizationType.Q4_0

    # Store hashes for all files and inline non-safetensors files as bytes.
    for name in file_names:
        p = model_dir / name
        writer.add_key_value(
            f"starvector.file_sha256.{name}",
            _sha256(p),
            gguf.GGUFValueType.STRING,
        )
        if not name.endswith(".safetensors"):
            raw = p.read_bytes()
            writer.add_key_value(
                f"starvector.file_bytes.{name}",
                list(raw),
                gguf.GGUFValueType.ARRAY,
                gguf.GGUFValueType.UINT8,
            )

    total = 0
    quantized_tensors = 0
    passthrough_tensors = 0
    unquantized_names: list[str] = []

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                arr = _numpy_from_reader(reader, key)
                if arr.dtype.kind not in ("f",):
                    # Candle gguf parser for quantized tensors expects ggml numeric types.
                    # Non-float buffers from safetensors (for example i64 batch counters) are
                    # stored as f32 weights here; original raw files are preserved in metadata.
                    arr = arr.astype(np.float32)

                if quant_type is not None and arr.dtype.kind in ("f",):
                    try:
                        qarr = gguf_quantize(arr.astype(np.float32, copy=False), quant_type)
                        writer.add_tensor(key, qarr, raw_dtype=quant_type)
                        quantized_tensors += 1
                    except Exception:
                        unquantized_names.append(key)
                        writer.add_tensor(key, arr)
                        passthrough_tensors += 1
                else:
                    writer.add_tensor(key, arr)
                    passthrough_tensors += 1
                total += 1

    if strict_quant and unquantized_names:
        sample = ", ".join(unquantized_names[:6])
        raise RuntimeError(
            f"Strict quantization failed for {len(unquantized_names)} tensors. Examples: {sample}"
        )

    writer.add_key_value(
        "starvector.quantized_tensor_count",
        quantized_tensors,
        gguf.GGUFValueType.UINT32,
    )
    writer.add_key_value(
        "starvector.passthrough_tensor_count",
        passthrough_tensors,
        gguf.GGUFValueType.UINT32,
    )
    if unquantized_names:
        writer.add_key_value(
            "starvector.unquantized_tensors",
            unquantized_names,
            gguf.GGUFValueType.ARRAY,
            gguf.GGUFValueType.STRING,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(
        f"Packed {total} tensors into {out_path} "
        f"(mode={quant_mode}, quantized={quantized_tensors}, passthrough={passthrough_tensors})"
    )


if __name__ == "__main__":
    main()
