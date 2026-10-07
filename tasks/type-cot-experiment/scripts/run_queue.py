"""Run Type-COT training, dev checkpoint selection, and test evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGE2_SCRIPTS = PROJECT_ROOT / "tasks/stage2-event-tree/scripts"
DEFAULT_MODEL = "/opt/models/Qwen3-8B"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(shlex.join(str(item) for item in command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_key(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def audit(data_path: Path, output_path: Path, model: str, max_len: int) -> dict:
    run([
        sys.executable,
        str(STAGE2_SCRIPTS / "train_lora.py"),
        "--data", str(data_path),
        "--output", str(output_path),
        "--model", model,
        "--max-len", str(max_len),
        "--audit-only",
    ], output_path.with_suffix(".log"))
    return read_json(output_path)


def evaluate_checkpoint(checkpoint: Path, split: str, args, run_dir: Path) -> tuple[dict, Path]:
    stem = f"{split}-{checkpoint.name}"
    prediction = run_dir / "checkpoint-eval" / f"{stem}.predictions.jsonl"
    metrics = run_dir / "checkpoint-eval" / f"{stem}.metrics.json"
    run([
        sys.executable,
        str(HERE / "predict.py"),
        "--data", str(args.data_dir / f"{split}.jsonl"),
        "--output", str(prediction),
        "--model", args.model,
        "--adapter", str(checkpoint),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--max-new-tokens", str(args.max_new_tokens),
    ], prediction.with_suffix(".predict.log"))
    run([
        sys.executable,
        str(HERE / "evaluate.py"),
        "--data", str(args.data_dir / f"{split}.jsonl"),
        "--predictions", str(prediction),
        "--output", str(metrics),
    ], metrics.with_suffix(".evaluate.log"))
    return read_json(metrics), prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "tasks/type-cot-experiment/data/type-cot-recognition")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1125)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "experiment": "type-cot-recognition",
        "status": "running",
        "started_at": now(),
        "seed": args.seed,
        "model": args.model,
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "max_len": args.max_len,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    save_json(run_dir / "run.json", state)
    try:
        train_audit = audit(args.data_dir / "train.jsonl", run_dir / "train.audit.json", args.model, args.max_len)
        state["audit"] = train_audit
        save_json(run_dir / "run.json", state)
        training_dir = run_dir / "training"
        run([
            sys.executable,
            str(STAGE2_SCRIPTS / "train_lora.py"),
            "--data", str(args.data_dir / "train.jsonl"),
            "--output", str(training_dir),
            "--model", args.model,
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--max-len", str(args.max_len),
            "--max-steps", str(args.max_steps),
        ], run_dir / "train.log")
        checkpoints = sorted((training_dir / "checkpoints").glob("checkpoint-*"), key=checkpoint_key)
        if not checkpoints:
            checkpoints = [training_dir / "adapter"]
        dev_results = []
        for checkpoint in checkpoints:
            metrics, prediction = evaluate_checkpoint(checkpoint, "dev", args, run_dir)
            dev_results.append({
                "checkpoint": str(checkpoint),
                "step": checkpoint_key(checkpoint),
                "type_multiset_f1": metrics["type_multiset"]["f1"],
                "metrics": str(run_dir / "checkpoint-eval" / f"dev-{checkpoint.name}.metrics.json"),
                "prediction": str(prediction),
            })
            state["dev_checkpoints"] = dev_results
            save_json(run_dir / "run.json", state)
        best = max(dev_results, key=lambda item: (item["type_multiset_f1"], item["step"]))
        best_checkpoint = Path(best["checkpoint"])
        test_metrics, test_prediction = evaluate_checkpoint(best_checkpoint, "test", args, run_dir)
        state.update({
            "status": "completed",
            "finished_at": now(),
            "best_checkpoint": str(best_checkpoint),
            "test_metrics": str(run_dir / "checkpoint-eval" / f"test-{best_checkpoint.name}.metrics.json"),
            "test_prediction": str(test_prediction),
            "test_type_multiset": test_metrics["type_multiset"],
            "test_parser_sources": test_metrics["parser_sources"],
        })
        save_json(run_dir / "run.json", state)
    except Exception as exc:
        state.update({"status": "failed", "failed_at": now(), "error": repr(exc)})
        save_json(run_dir / "run.json", state)
        raise


if __name__ == "__main__":
    main()
