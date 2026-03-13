import argparse
import json
import sys
from pathlib import Path


def _import_error(name: str, exc: Exception) -> str:
    return f"{name}: {type(exc).__name__}: {exc}"


def _candidate_starvector_paths(model_dir: Path) -> list[Path]:
    candidates = []
    env = Path.cwd()
    candidates.append(env)
    candidates.append(model_dir.parent.parent / "star-vector")
    return candidates


def preflight_mode(model_dir: Path) -> int:
    required = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("Pillow", "PIL"),
        ("safetensors", "safetensors"),
    ]
    missing = []
    for pkg, module_name in required:
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(_import_error(pkg, exc))

    starvector_ok = False
    starvector_error = None
    try:
        __import__("starvector")
        starvector_ok = True
    except Exception as exc:
        # Try explicit local fallback path.
        for candidate in _candidate_starvector_paths(model_dir):
            if candidate.exists():
                sys.path.insert(0, str(candidate))
                try:
                    __import__("starvector")
                    starvector_ok = True
                    starvector_error = None
                    break
                except Exception as retry_exc:
                    starvector_error = _import_error("starvector", retry_exc)
        if not starvector_ok and starvector_error is None:
            starvector_error = _import_error("starvector", exc)

    payload = {
        "ok": (len(missing) == 0 and starvector_ok),
        "missing_modules": missing,
        "starvector_import_ok": starvector_ok,
        "starvector_error": starvector_error,
        "bootstrap_hint": (
            "py -3 -m pip install torch torchvision transformers tokenizers Pillow safetensors "
            "and set PYTHONPATH=D:\\starvecntor\\star-vector"
        ),
    }
    print(json.dumps(payload))
    return 0 if payload["ok"] else 2


def tokenize_mode(tokenizer_json: Path, texts: list[str]) -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    results = []
    for text in texts:
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids, False)
        results.append({"text": text, "ids": enc.ids, "decoded": dec})
    print(json.dumps({"results": results}, ensure_ascii=False))


def preprocess_mode(model_dir: Path, image_path: Path) -> None:
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode, pad

    processor_config = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    size = int(processor_config["size"])
    mean = tuple(float(x) for x in processor_config["mean"])
    std = tuple(float(x) for x in processor_config["std"])

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode == "RGBA" else img),
            transforms.Lambda(lambda img: _pad_to_square(img, fill=255)),
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )

    image = Image.open(image_path)
    tensor = transform(image).cpu().numpy().astype(np.float32, copy=False)
    payload = {"shape": list(tensor.shape), "values": tensor.reshape(-1).tolist()}
    print(json.dumps(payload))


def adapter_mode(model_dir: Path, input_json_path: Path) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    shape = tuple(int(v) for v in payload["shape"])
    values = payload["values"]
    x = torch.tensor(values, dtype=torch.float32).reshape(shape)

    shard = model_dir / "model-00002-of-00002.safetensors"
    weights = load_file(str(shard))

    c_fc_w = weights["model.image_projection.c_fc.weight"].float()
    c_fc_b = weights["model.image_projection.c_fc.bias"].float()
    c_proj_w = weights["model.image_projection.c_proj.weight"].float()
    c_proj_b = weights["model.image_projection.c_proj.bias"].float()

    bn_weight = weights["model.image_projection.norm.weight"].float()
    bn_bias = weights["model.image_projection.norm.bias"].float()
    bn_running_mean = weights["model.image_projection.norm.running_mean"].float()
    bn_running_var = weights["model.image_projection.norm.running_var"].float()
    bn_tracked = int(weights["model.image_projection.norm.num_batches_tracked"].item())

    y = F.linear(x, c_fc_w, c_fc_b)
    y = y * torch.sigmoid(y)
    y = F.linear(y, c_proj_w, c_proj_b)
    y = F.batch_norm(
        y,
        running_mean=bn_running_mean,
        running_var=bn_running_var,
        weight=bn_weight,
        bias=bn_bias,
        training=False,
        momentum=0.1,
        eps=1e-5,
    )

    out = y.detach().cpu().numpy().astype(np.float32, copy=False)
    result = {
        "shape": list(out.shape),
        "values": out.reshape(-1).tolist(),
        "num_batches_tracked": bn_tracked,
    }
    print(json.dumps(result))


def _load_decoder_state_dict(model_dir: Path) -> dict:
    from safetensors.torch import safe_open

    state = {}
    prefix = "model.svg_transformer.transformer."
    for shard_name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        shard_path = model_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = reader.get_tensor(key).float()
    return state


def greedy_step_trace_mode(
    model_dir: Path,
    input_json_path: Path,
    max_new_tokens: int,
    device_name: str,
    tokenizer_json: Path | None,
) -> None:
    import torch
    from tokenizers import Tokenizer
    from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    shape = tuple(int(v) for v in payload["shape"])
    values = payload["values"]
    stop_ids = [int(v) for v in payload["stop_ids"]]

    config_json = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = GPTBigCodeConfig(
        vocab_size=int(config_json["vocab_size"]),
        n_positions=int(config_json["max_position_embeddings"]),
        n_layer=int(config_json["num_hidden_layers"]),
        n_head=int(config_json["num_attention_heads"]),
        n_embd=int(config_json["hidden_size"]),
        n_inner=None,
        multi_query=bool(config_json["multi_query"]),
        use_cache=bool(config_json.get("use_cache", True)),
        layer_norm_epsilon=1e-5,
    )

    model = GPTBigCodeForCausalLM(cfg)
    state_dict = _load_decoder_state_dict(model_dir)
    if "transformer.wte.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"decoder load missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"decoder load unexpected keys: {unexpected}")
    model.eval()

    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")
    model.to(device)

    prefix_embeds = torch.tensor(values, dtype=torch.float32, device=device).reshape(shape)

    generated = []
    with torch.no_grad():
        out = model.transformer(
            inputs_embeds=prefix_embeds, use_cache=True, return_dict=True
        )
        logits = model.lm_head(out.last_hidden_state[:, -1, :])
        past_key_values = out.past_key_values

        for _ in range(max_new_tokens):
            next_token = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_token)
            if len(generated) >= len(stop_ids) and generated[-len(stop_ids) :] == stop_ids:
                break
            token = torch.tensor([[next_token]], dtype=torch.long, device=device)
            out = model.transformer(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = model.lm_head(out.last_hidden_state[:, -1, :])
            past_key_values = out.past_key_values

    result = {"token_ids": generated}
    if tokenizer_json is not None:
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        result["decoded"] = tokenizer.decode(generated, False)
    print(json.dumps(result))


def _pad_to_square(img, fill=255):
    from torchvision.transforms.functional import pad

    width, height = img.size
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    right = max_dim - width - left
    bottom = max_dim - height - top
    return pad(img, [left, top, right, bottom], fill=fill)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["preflight", "tokenize", "preprocess", "adapter", "greedy_step_trace"],
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--text", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.mode == "preflight":
        if args.model_dir is None:
            raise ValueError("--model-dir is required for preflight mode")
        return preflight_mode(args.model_dir)

    if args.mode == "tokenize":
        if args.tokenizer_json is None:
            raise ValueError("--tokenizer-json is required for tokenize mode")
        if not args.text:
            raise ValueError("at least one --text is required for tokenize mode")
        tokenize_mode(args.tokenizer_json, args.text)
        return 0

    if args.mode == "preprocess":
        if args.model_dir is None or args.image is None:
            raise ValueError("--model-dir and --image are required for preprocess mode")
        preprocess_mode(args.model_dir, args.image)
        return 0

    if args.mode == "adapter":
        if args.model_dir is None or args.input_json is None:
            raise ValueError("--model-dir and --input-json are required for adapter mode")
        adapter_mode(args.model_dir, args.input_json)
        return 0

    if args.mode == "greedy_step_trace":
        if args.model_dir is None or args.input_json is None:
            raise ValueError("--model-dir and --input-json are required for greedy_step_trace mode")
        greedy_step_trace_mode(
            args.model_dir,
            args.input_json,
            args.max_new_tokens,
            args.device,
            args.tokenizer_json,
        )
        return 0

    raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"python_oracle_error: {exc}", file=sys.stderr)
        raise
