"""Build a Type-COT SFT dataset from merged prompt-3.10 COT records."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from parser import EVENT_CODES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COT_TRAIN = PROJECT_ROOT / "task/prompt/runs/prompt-3.10-1-train-full-merged-20260926.jsonl"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "tasks/stage2-event-tree/data/sentence-c"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tasks/type-cot-experiment/data/type-cot-recognition"


TYPE_DESCRIPTIONS = {
    "PUR": "Purpose: research aim or intended objective.",
    "ITT": "Background/IntroduceTarget: abstract research target or task definition.",
    "RWS": "RelatedWorkStep: prior/existing procedure, component, or approach.",
    "RWF": "RelatedWorkFault: limitation, deficiency, or fault motivating the work.",
    "PRP": "Propose: authors introduce/propose a method, model, dataset, or framework.",
    "WKS": "WorkStatement: authors report or state work content, settings, or study actions.",
    "MDS": "MethodStep: component-assisted operation inside the proposed method.",
    "FIN": "Finding: experimental finding or demonstrated conclusion.",
    "CMP": "Comparison: explicit comparison between systems/results/baselines.",
    "FAC": "ResultEffect: subject provides/improves/enables an outcome or capability.",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sample_id_to_source_id(sample_id: str) -> str:
    split, doc_id, sentence_index = sample_id.split(":", 2)
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"unexpected split in sample_id: {sample_id}")
    return f"{doc_id}::sentence-{int(sentence_index)}"


def event_type_inventory(events: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events:
        event_type = event.get("event_type")
        if event_type not in EVENT_CODES:
            raise ValueError(f"unknown event type: {event_type!r}")
        counts[event_type] += 1
    return counts


def inventory_text(counts: Counter[str]) -> str:
    lines = [f"{code}: {counts[code]}" for code in EVENT_CODES if counts[code]]
    return "\n".join(lines) if lines else "None"


def parse_gold_inventory(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not text or text.strip() == "None":
        return counts
    for code, count in re.findall(r"\b([A-Z]{3})\s*:\s*(\d+)", text):
        if code not in EVENT_CODES:
            raise ValueError(f"unknown inventory type: {code}")
        counts[code] += int(count)
    return counts


def make_prompt(source_row: dict) -> str:
    schema = "\n".join(f"- {code}: {TYPE_DESCRIPTIONS[code]}" for code in EVENT_CODES)
    marked_sentences = "\n".join(
        f"[S{i + 1}] {sentence}" for i, sentence in enumerate(source_row.get("sentences", []))
    )
    return "\n\n".join(
        [
            "You are an expert in scientific event extraction.",
            "Given a full scientific abstract and one TARGET SENTENCE, identify all event types "
            "expressed by triggers in the target sentence. A sentence may contain multiple events "
            "and repeated event types.",
            "Think through the semantic evidence first, then output a final Event Type Inventory. "
            "Do not extract trigger spans or arguments.",
            "Allowed event types:\n" + schema,
            "Full Abstract:\n" + (marked_sentences or source_row["abstract"]),
            f"TARGET SENTENCE [S{source_row['sentence_index'] + 1}]:\n{source_row['target_sentence']}",
            "Type-COT output:",
        ]
    )


def completion_with_inventory(reasoning: str, counts: Counter[str]) -> str:
    body = reasoning.strip()
    final = "Final Answer:\nEvent Type Inventory:\n" + inventory_text(counts)
    if "Event Type Inventory:" in body:
        return body
    return body + "\n\n" + final if body else final


def _base_output(source_row: dict) -> dict:
    keep = [
        "id", "document_id", "split", "mode", "sentence_index", "target_bounds",
        "target_sentence", "tokens", "sentences", "abstract", "target_events", "n_events",
    ]
    output = {key: source_row[key] for key in keep if key in source_row}
    output["mode"] = "type-cot-recognition"
    output["prompt"] = make_prompt(source_row)
    counts = event_type_inventory(source_row.get("target_events", []))
    output["target_type_inventory"] = {code: counts[code] for code in EVENT_CODES if counts[code]}
    return output


def build_train_row(cot_row: dict, source_row: dict) -> dict:
    output = _base_output(source_row)
    gold_counts = parse_gold_inventory(cot_row.get("gold_event_type_inventory", ""))
    source_counts = event_type_inventory(source_row.get("target_events", []))
    if gold_counts != source_counts:
        raise ValueError(f"{cot_row.get('sample_id')}: COT inventory differs from source events")
    output["completion"] = completion_with_inventory(cot_row.get("completion", ""), source_counts)
    output["source_sample_id"] = cot_row["sample_id"]
    output["cot_api_used"] = cot_row.get("api_used")
    output["cot_qc"] = cot_row.get("qc", {})
    return output


def build_eval_row(source_row: dict) -> dict:
    output = _base_output(source_row)
    output["completion"] = "Final Answer:\nEvent Type Inventory:\n" + inventory_text(
        event_type_inventory(source_row.get("target_events", []))
    )
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_all(cot_train: Path, source_dir: Path, output_dir: Path) -> dict:
    cot_rows = read_jsonl(cot_train)
    source = {split: read_jsonl(source_dir / f"{split}.jsonl") for split in ("train", "dev", "test")}
    source_train_by_id = {row["id"]: row for row in source["train"]}
    train_rows: list[dict] = []
    skipped = Counter()
    for cot_row in cot_rows:
        if not cot_row.get("completed") or not cot_row.get("qc", {}).get("ok"):
            skipped["not_completed_or_bad_qc"] += 1
            continue
        source_id = sample_id_to_source_id(cot_row["sample_id"])
        if source_id not in source_train_by_id:
            skipped["missing_source"] += 1
            continue
        train_rows.append(build_train_row(cot_row, source_train_by_id[source_id]))
    split_rows = {
        "train": train_rows,
        "dev": [build_eval_row(row) for row in source["dev"]],
        "test": [build_eval_row(row) for row in source["test"]],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": now(),
        "mode": "type-cot-recognition",
        "cot_train": str(cot_train.resolve()),
        "cot_train_sha256": sha256(cot_train),
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "leakage_policy": "prompts exclude gold_event_type_inventory and gold_arguments_by_event",
        "splits": {},
        "skipped_train": dict(skipped),
    }
    for split, rows in split_rows.items():
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        event_counts = Counter(
            event["event_type"] for row in rows for event in row.get("target_events", [])
        )
        manifest["splits"][split] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(rows),
            "events": sum(event_counts.values()),
            "event_type_counts": dict(event_counts),
            "none_rows": sum(1 for row in rows if not row.get("target_events")),
            "max_completion_words": max((len(row["completion"].split()) for row in rows), default=0),
        }
    manifest["checks"] = {
        "train_rows_match_completed_cot": len(train_rows) + sum(skipped.values()) == len(cot_rows),
        "train_prompt_has_no_gold_inventory_token": not any(
            "gold_event_type_inventory" in row["prompt"] or "gold_arguments_by_event" in row["prompt"]
            for row in train_rows
        ),
    }
    manifest["checks"]["all_pass"] = all(manifest["checks"].values())
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cot-train", type=Path, default=DEFAULT_COT_TRAIN)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build_all(args.cot_train, args.source_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
