from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def _add_llama_cpp_gguf_to_path(repo_root: Path) -> None:
    gguf_py = repo_root / "references" / "llama.cpp" / "gguf-py"
    if not gguf_py.exists():
        raise FileNotFoundError(f"gguf-py not found at {gguf_py}")
    sys.path.insert(0, str(gguf_py))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_field(reader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise KeyError(f"Missing GGUF metadata field: {key}")
    return field


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify full StarVector GGUF package metadata against source model directory."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    model_dir = args.model_dir.resolve()
    gguf_path = args.gguf.resolve()

    _add_llama_cpp_gguf_to_path(repo_root)
    from gguf.gguf_reader import GGUFReader  # type: ignore

    reader = GGUFReader(str(gguf_path))
    files = list(_require_field(reader, "starvector.files").contents())
    local_files = sorted(p.name for p in model_dir.iterdir() if p.is_file())

    errors: list[str] = []
    if sorted(files) != local_files:
        errors.append("File list mismatch between GGUF metadata and model directory")

    verified_hashes = 0
    verified_embedded = 0

    for name in files:
        src_path = model_dir / name
        if not src_path.exists():
            errors.append(f"Missing source file: {name}")
            continue

        hash_key = f"starvector.file_sha256.{name}"
        expected_hash = str(_require_field(reader, hash_key).contents())
        actual_hash = _sha256(src_path)
        if expected_hash != actual_hash:
            errors.append(f"SHA256 mismatch for {name}")
        else:
            verified_hashes += 1

        if not name.endswith(".safetensors"):
            bytes_key = f"starvector.file_bytes.{name}"
            embedded_list = _require_field(reader, bytes_key).contents()
            embedded = bytes(embedded_list)
            actual = src_path.read_bytes()
            if embedded != actual:
                errors.append(f"Embedded bytes mismatch for {name}")
            else:
                verified_embedded += 1

    if errors:
        print("Verification FAILED:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print(
        "Verification OK: "
        f"{len(files)} files listed, "
        f"{verified_hashes} hashes matched, "
        f"{verified_embedded} embedded files matched byte-for-byte."
    )


if __name__ == "__main__":
    main()
