"""Check pinned CPU imports without loading models or creating CUDA tensors."""
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    expected = json.loads(Path(__file__).with_name("runtime.json").read_text())
    names = ["torch", "transformers", "peft", "accelerate", "numpy", "scipy",
             "safetensors", "tokenizers"]
    actual = {}
    mismatches = []
    for name in names:
        importlib.import_module(name)
        actual[name] = importlib.metadata.version(name)
        if actual[name] != expected["versions"][name]:
            mismatches.append(name)
    import torch
    from scipy.optimize import linear_sum_assignment
    from transformers import GenerationConfig
    row, col = linear_sum_assignment([[1, 0], [0, 1]])
    assert list(row) == [0, 1] and list(col) == [1, 0]
    assert GenerationConfig().temperature == 1.0
    info = {"versions": actual, "torch_cuda": torch.version.cuda,
            "python": platform.python_version(), "machine": platform.machine(),
            "libc": platform.libc_ver(), "version_mismatches": mismatches}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if torch.version.cuda != expected["torch_cuda"]:
        raise SystemExit("CUDA runtime differs from exported environment")
    if mismatches:
        raise SystemExit("Package versions differ from exported environment")


if __name__ == "__main__":
    main()
