"""Evaluate Type-COT predictions as event-type multisets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from parser import EVENT_CODES, parse_type_cot_output


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _gold_types(row: dict) -> list[str]:
    return [event["event_type"] for event in row.get("target_events", [])]


def _multiset_scores(gold: list[str], pred: list[str]) -> dict:
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)
    tp = sum(min(gold_counts[code], pred_counts[code]) for code in EVENT_CODES)
    fp = sum(max(0, pred_counts[code] - gold_counts[code]) for code in EVENT_CODES)
    fn = sum(max(0, gold_counts[code] - pred_counts[code]) for code in EVENT_CODES)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _prf_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_rows(data_rows: list[dict], predictions: list[dict]) -> dict:
    pred_by_id = {row["id"]: row for row in predictions}
    if len(pred_by_id) != len(predictions):
        raise ValueError("duplicate prediction ids")
    data_ids = {row["id"] for row in data_rows}
    if set(pred_by_id) != data_ids:
        raise ValueError("prediction ids do not exactly match data ids")
    aggregate = Counter()
    per_type = {code: Counter() for code in EVENT_CODES}
    parser_sources = Counter()
    for row in data_rows:
        parsed = pred_by_id[row["id"]].get("parse") or parse_type_cot_output(
            pred_by_id[row["id"]].get("raw_output", "")
        )
        parser_sources[parsed["source"]] += 1
        gold = _gold_types(row)
        pred = parsed["types"]
        score = _multiset_scores(gold, pred)
        aggregate.update({key: score[key] for key in ("tp", "fp", "fn")})
        gold_counts = Counter(gold)
        pred_counts = Counter(pred)
        for code in EVENT_CODES:
            per_type[code]["gold"] += gold_counts[code]
            per_type[code]["pred"] += pred_counts[code]
            per_type[code]["tp"] += min(gold_counts[code], pred_counts[code])
            per_type[code]["fp"] += max(0, pred_counts[code] - gold_counts[code])
            per_type[code]["fn"] += max(0, gold_counts[code] - pred_counts[code])
    result = _prf_from_counts(aggregate["tp"], aggregate["fp"], aggregate["fn"])
    for code, counts in per_type.items():
        prf = _prf_from_counts(counts["tp"], counts["fp"], counts["fn"])
        counts.update({key: prf[key] for key in ("precision", "recall", "f1")})
    return {
        "rows": len(data_rows),
        "type_multiset": result,
        "per_type": {code: dict(counts) for code, counts in per_type.items()},
        "parser_sources": dict(parser_sources),
    }


def evaluate_files(data_path: Path, predictions_path: Path, output_path: Path) -> dict:
    result = evaluate_rows(read_jsonl(data_path), read_jsonl(predictions_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_files(args.data, args.predictions, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
