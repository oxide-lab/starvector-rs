from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export StarVector decoder weights to GPTBigCode HF format for GGUF conversion."
    )
    parser.add_argument("--src-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    src = args.src_model_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    src_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg = {
        "architectures": ["GPTBigCodeForCausalLM"],
        "model_type": "gpt_bigcode",
        "vocab_size": int(src_cfg["vocab_size"]),
        "n_positions": int(src_cfg["max_position_embeddings"]),
        "n_embd": int(src_cfg["hidden_size"]),
        "n_layer": int(src_cfg["num_hidden_layers"]),
        "n_head": int(src_cfg["num_attention_heads"]),
        "layer_norm_epsilon": 1e-5,
        "multi_query": bool(src_cfg.get("multi_query", True)),
        "use_cache": bool(src_cfg.get("use_cache", True)),
        "bos_token_id": 0,
        "eos_token_id": 0,
        "tie_word_embeddings": True,
        "transformers_version": "4.40.1",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
    ):
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, out / name)

    prefix = "model.svg_transformer.transformer."
    weights: dict[str, torch.Tensor] = {}
    shard = src / "model-00001-of-00002.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                mapped = key[len(prefix) :]
                weights[mapped] = f.get_tensor(key).to(torch.float16)

    if "transformer.wte.weight" in weights and "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["transformer.wte.weight"].clone()

    save_file(weights, str(out / "model.safetensors"))
    index = {
        "metadata": {
            "total_size": sum(t.numel() * t.element_size() for t in weights.values())
        },
        "weight_map": {k: "model.safetensors" for k in weights.keys()},
    }
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(weights)} tensors into {out}")


if __name__ == "__main__":
    main()
