#!/usr/bin/env python3
"""Analyze original Experiment C difficulty slices without model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "tasks/stage2-event-tree"
sys.path.insert(0, str(STAGE2 / "scripts"))
sys.path.insert(0, str(ROOT))

from align_events import sentence_bounds  # noqa: E402
from evaluate import METRIC_NAMES, official_scores  # noqa: E402

EVENT_TYPES = ("PUR", "ITT", "RWS", "RWF", "PRP", "WKS", "MDS", "FIN", "CMP", "FAC")
ROLE_ID = {"Researcher": "OG", "Proposer": "OG", "Finder": "OG"}
METRIC_KEYS = ("TI_f1", "TC_f1", "AI_f1", "AC_f1")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_schema(path):
    text = Path(path).read_text(encoding="utf-8")
    schemas = {}
    for match in re.finditer(r"^### ([A-Z]+) — ([^\n]+)\n\n```text\n.*?\n```\n\nRoles:\n\n```text\n(.*?)\n```", text, re.M | re.S):
        code, full_name, role_block = match.groups()
        roles = [line.strip() for line in role_block.splitlines() if line.strip()]
        schemas[code] = {"full_name": full_name.strip(), "roles": roles}
    if tuple(schemas) != EVENT_TYPES:
        raise ValueError(f"Appendix schema mismatch/order: {tuple(schemas)}")
    return schemas


def argument_geometry(offsets):
    """Return geometric span properties; offsets are token positions, not text order."""
    if not offsets:
        return {"fragment_count": None, "gap_tokens": None, "max_gap_tokens": None,
                "contiguous": None, "discontinuous": None}
    positions = sorted(set(int(x) for x in offsets))
    runs = []
    start = previous = positions[0]
    gaps = []
    for position in positions[1:]:
        if position != previous + 1:
            runs.append((start, previous))
            gaps.append(position - previous - 1)
            start = position
        previous = position
    runs.append((start, previous))
    return {"fragment_count": len(runs), "gap_tokens": sum(gaps),
            "max_gap_tokens": max(gaps, default=0), "contiguous": len(runs) == 1,
            "discontinuous": len(runs) > 1}


def schema_similarity(observed, schema_roles, role_weights):
    observed, schema_roles = set(observed), set(schema_roles)
    union = observed | schema_roles
    if not union:
        return 0.0
    denominator = sum(role_weights.get(role, 1.0) for role in union)
    return (sum(role_weights.get(role, 1.0) for role in observed & schema_roles) / denominator
            if denominator else 0.0)


def multiset_scores(gold_types, pred_types):
    gold, pred = Counter(gold_types), Counter(pred_types)
    tp = sum((gold & pred).values())
    fp = sum((pred - gold).values())
    fn = sum((gold - pred).values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "exact": gold == pred}


def _span(event, key):
    value = event.get(key, {})
    return tuple(sorted(set(value.get("offsets", []))))


def _trigger_text(event):
    return " ".join(event.get("trigger", {}).get("text", "").casefold().split())


def _arg_set(event):
    return {(a.get("argument_type", a.get("role", "")), tuple(sorted(set(a.get("offsets", [])))))
            for a in event.get("arguments", event.get("args", [])) if a.get("offsets")}


def _f1_overlap(left, right):
    left, right = set(left), set(right)
    return 2 * len(left & right) / (len(left) + len(right)) if left and right else 0.0


def _event_pair_weight(gold, pred):
    gs, ps = _span(gold, "trigger"), _span(pred, "trigger")
    if gs and ps and gs == ps:
        tier = 4
    elif gs and ps and set(gs) & set(ps):
        tier = 3
    elif _trigger_text(gold) and _trigger_text(gold) == _trigger_text(pred):
        tier = 2
    else:
        tier = 0
    arg_gold, arg_pred = _arg_set(gold), _arg_set(pred)
    arg_score = _f1_overlap(arg_gold, arg_pred)
    if tier == 0 and arg_score > 0:
        tier = 1
    if tier == 0:
        return 0.0, "none"
    tier_name = {1: "argument-assisted", 2: "trigger-text", 3: "trigger-overlap", 4: "trigger-exact"}[tier]
    trigger_score = _f1_overlap(gs, ps)
    return tier * 1_000_000 + trigger_score * 1_000 + arg_score * 10, tier_name


def _hungarian_max(weights):
    """Maximum-weight assignment for rows <= columns (integer-safe Hungarian)."""
    n = len(weights)
    if not n:
        return []
    m = len(weights[0])
    if n > m:
        raise ValueError("Hungarian matrix must have at least as many columns as rows")
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    u, v = [0] * (n + 1), [0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [10**30] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = 10**30, 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j], way[j] = current, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def match_events(gold, pred):
    """Type-agnostic one-to-one matching; trigger evidence first, args fallback."""
    if not gold:
        return [], [], list(range(len(pred)))
    weights, tiers = [], []
    for ge in gold:
        row, tier_row = [], []
        for pe in pred:
            weight, tier = _event_pair_weight(ge, pe)
            row.append(int(weight * 1000))
            tier_row.append(tier)
        # One zero-weight dummy per gold event allows unmatched rows.
        row.extend([0] * len(gold))
        tier_row.extend(["unmatched"] * len(gold))
        weights.append(row)
        tiers.append(tier_row)
    assignment = _hungarian_max(weights)
    matches, missed, used_pred = [], [], set()
    for gi, col in enumerate(assignment):
        if 0 <= col < len(pred) and weights[gi][col] > 0:
            matches.append((gi, col, tiers[gi][col]))
            used_pred.add(col)
        else:
            missed.append(gi)
    spurious = [pi for pi in range(len(pred)) if pi not in used_pred]
    return matches, missed, spurious


def _csv_list(values):
    return "|".join(str(value) for value in values)


def _role_statistics(train_rows, schemas):
    per_type = {event_type: Counter() for event_type in EVENT_TYPES}
    totals = Counter()
    for row in train_rows:
        for event in row["target_events"]:
            event_type = event["event_type"]
            if event_type not in schemas:
                raise ValueError(f"Unknown gold type {event_type}")
            totals[event_type] += 1
            per_type[event_type].update(set(a["argument_type"] for a in event["arguments"]))
    role_df = Counter(role for schema in schemas.values() for role in set(schema["roles"]))
    idf = {role: math.log(len(EVENT_TYPES) / count) for role, count in role_df.items()}
    p_role = {event_type: {role: per_type[event_type][role] / totals[event_type]
                           for role in schemas[event_type]["roles"]}
              for event_type in EVENT_TYPES}
    role_weights = {role: idf.get(role, 0.0) for role in idf}
    combined = {event_type: {role: p_role[event_type][role] * idf.get(role, 0.0)
                             for role in schemas[event_type]["roles"]}
                for event_type in EVENT_TYPES}
    return {"totals": totals, "p_role": p_role, "idf": idf,
            "role_weights": role_weights, "combined": combined}


def _doc_argument_relations(rows):
    by_doc = defaultdict(list)
    event_span_counts = {}
    for row in rows:
        for ei, event in enumerate(row["target_events"]):
            event_key = (row["document_id"], row["id"], ei)
            local = Counter(tuple(sorted(set(arg.get("offsets", [])))) for arg in event["arguments"]
                            if arg.get("offsets"))
            event_span_counts[event_key] = local
            for ai, arg in enumerate(event["arguments"]):
                span = tuple(sorted(set(arg.get("offsets", []))))
                if span:
                    by_doc[row["document_id"]].append((span, event_key, ai))
    nested_events, shared_events = set(), set()
    for entries in by_doc.values():
        for i, (left, left_key, _) in enumerate(entries):
            left_set = set(left)
            for right, right_key, _ in entries[i + 1:]:
                right_set = set(right)
                if left_key != right_key and left == right:
                    shared_events.update((left_key, right_key))
                if left != right and (left_set < right_set or right_set < left_set):
                    nested_events.update((left_key, right_key))
    same_span_events = {key for key, counts in event_span_counts.items() if any(count > 1 for count in counts.values())}
    return same_span_events, nested_events, shared_events


def _event_features(row, event, index, schema, stats, relations):
    event_type = event["event_type"]
    allowed = schema[event_type]["roles"]
    observed = set(arg["argument_type"] for arg in event["arguments"])
    missing = [role for role in allowed if role not in observed]
    plain = len(observed & set(allowed)) / len(allowed) if allowed else None
    weights = stats["combined"][event_type]
    denominator = sum(weights.values())
    recoverability = (sum(weights.get(role, 0.0) for role in observed) / denominator
                      if denominator else plain)
    similarities = {candidate: schema_similarity(observed, schema[candidate]["roles"], stats["idf"])
                    for candidate in EVENT_TYPES}
    gold_score = similarities[event_type]
    competing = sorted(((score, candidate) for candidate, score in similarities.items()
                        if candidate != event_type), key=lambda item: (-item[0], item[1]))
    best_score, best_type = competing[0]
    lo, hi = row["target_bounds"]
    trigger_offsets = set(event["trigger"]["offsets"])
    sent_bounds = sentence_bounds(row["sentences"])
    span_stats, reverse = [], False
    for arg in event["arguments"]:
        offsets = arg.get("offsets", [])
        span_stats.append(argument_geometry(offsets))
        if offsets and trigger_offsets and max(offsets) < min(trigger_offsets):
            reverse = True
    has_cross = any(any(not (lo <= offset < hi) for offset in arg.get("offsets", []))
                    for arg in event["arguments"])
    event_key = (row["document_id"], row["id"], index)
    same_span_events, nested_events, shared_events = relations
    valid_geometries = [value for value in span_stats if value["fragment_count"] is not None]
    max_fragments = max((value["fragment_count"] for value in valid_geometries), default=None)
    max_gap = max((value["max_gap_tokens"] for value in valid_geometries), default=None)
    return {
        "sample_id": row["id"], "abstract_id": row["document_id"],
        "target_sentence_id": row["sentence_index"], "gold_event_index": index,
        "gold_event_type": event_type, "trigger_text": event["trigger"]["text"],
        "num_arguments": len(event["arguments"]), "observed_roles": _csv_list(sorted(observed)),
        "missing_roles": _csv_list(missing), "plain_role_coverage": plain,
        "weighted_schema_recoverability": recoverability, "gold_schema_score": gold_score,
        "best_competing_type": best_type, "best_competing_score": best_score,
        "schema_margin": gold_score - best_score,
        "has_cross_sentence_argument": has_cross,
        "has_same_span_reuse": event_key in same_span_events,
        "has_nested": event_key in nested_events, "has_shared": event_key in shared_events,
        "has_discontinuous": any(value["discontinuous"] for value in valid_geometries),
        "has_reverse": reverse, "max_num_fragments": max_fragments,
        "max_gap_tokens": max_gap,
        "has_normal_contiguous_argument": any(value["contiguous"] for value in valid_geometries),
        "span_geometries": span_stats,
    }


def _quantile_labels(samples, field, label_suffix=""):
    values = sorted(float(sample[field]) for sample in samples if sample.get(field) is not None)
    if not values:
        return {}
    # Empirical tertiles. Ties remain together; buckets can be uneven by design.
    lower = values[max(0, math.ceil(len(values) / 3) - 1)]
    upper = values[max(0, math.ceil(2 * len(values) / 3) - 1)]
    output = {}
    for sample in samples:
        value = sample.get(field)
        if value is None:
            continue
        value = float(value)
        if lower == upper:
            label = "Medium"
        elif value <= lower:
            label = "Low"
        elif value <= upper:
            label = "Medium"
        else:
            label = "High"
        output[sample["sample_id"]] = label + label_suffix
    return output


def _type_only_for_samples(samples):
    totals = Counter()
    exact = 0
    for sample in samples:
        scores = multiset_scores(sample["gold_types"], sample["pred_types"])
        totals.update({key: scores[key] for key in ("tp", "fp", "fn")})
        exact += scores["exact"]
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"samples": len(samples), "exact": exact / len(samples) if samples else None,
            "type_p": p, "type_r": r, "type_f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _official_for_samples(samples):
    if not samples:
        return {key: None for key in METRIC_KEYS}
    score = official_scores([sample["gold_events"] for sample in samples],
                            [sample["pred_events"] for sample in samples])
    return {key: score[key] for key in METRIC_KEYS}


def _slice_record(label, samples):
    return {"slice": label, **_type_only_for_samples(samples), **_official_for_samples(samples)}


def _spearman(x, y):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + 1 + end) / 2
            for k in range(start, end):
                out[order[k]] = rank
            start = end
        return out
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def _schema_similarity_matrix(schemas, idf):
    rows, matrix = [], {}
    for left in EVENT_TYPES:
        matrix[left] = {}
        for right in EVENT_TYPES:
            score = schema_similarity(schemas[left]["roles"], schemas[right]["roles"], idf)
            matrix[left][right] = score
        rows.append({"event_type": left, **matrix[left]})
    pairs = []
    for i, left in enumerate(EVENT_TYPES):
        for right in EVENT_TYPES[i + 1:]:
            pairs.append({"type_a": left, "type_b": right,
                          "weighted_similarity": matrix[left][right],
                          "unweighted_jaccard": schema_similarity(schemas[left]["roles"], schemas[right]["roles"],
                                                                   {role: 1.0 for role in idf})})
    pairs.sort(key=lambda row: (-row["weighted_similarity"], row["type_a"], row["type_b"]))
    return rows, matrix, pairs


def _normalize_trigger_lexicon(train_rows):
    return {tuple(token.casefold() for token in event["trigger"]["tokens"])
            for row in train_rows for event in row["target_events"] if event["trigger"]["tokens"]}


def _trigger_like_count(sentence, lexicon):
    tokens = [token.casefold() for token in sentence.split()]
    return sum(1 for start in range(len(tokens)) for trigger in lexicon
               if tokens[start:start + len(trigger)] == list(trigger))


def _complexity_categories(feature):
    categories = []
    if feature["num_arguments"] and all(g["contiguous"] for g in feature["span_geometries"] if g["fragment_count"]):
        categories.append("normal_contiguous")
    for key, label in (("has_same_span_reuse", "same_span_reuse"), ("has_nested", "nested"),
                       ("has_shared", "shared"), ("has_reverse", "reverse"),
                       ("has_cross_sentence_argument", "cross_sentence")):
        if feature[key]:
            categories.append(label)
    for geometry in feature["span_geometries"]:
        if not geometry["discontinuous"]:
            continue
        if geometry["fragment_count"] == 2:
            gap = geometry["max_gap_tokens"]
            categories.append("discontinuous_2frag_gap_le_3" if gap <= 3 else
                              "discontinuous_2frag_gap_4_102" if gap <= 102 else
                              "discontinuous_2frag_gap_gt_102")
        elif geometry["fragment_count"] >= 3:
            categories.append("discontinuous_3plus")
    return sorted(set(categories))


def _report_table(rows, columns):
    if not rows:
        return "(no samples)"
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if value is None:
                values.append("NA")
            elif isinstance(value, float):
                values.append(f"{value * 100:.2f}%" if col.endswith(("f1", "_p", "_r", "exact", "type_p", "type_r", "type_f1", "_rate")) or col in ("TI_f1", "TC_f1", "AI_f1", "AC_f1") else f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *body])


def analyze(root=ROOT, output_dir=None):
    root = Path(root).resolve()
    output_dir = Path(output_dir or root / "tasks/stage2-event-tree/difficulty-analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(root / "tasks/stage2-event-tree/data/sentence-c/train.jsonl")
    dev_rows = read_jsonl(root / "tasks/stage2-event-tree/data/sentence-c/dev.jsonl")
    pred_rows = read_jsonl(root / "tasks/stage2-event-tree/runs/C-42-20260918/checkpoint-eval/dev-checkpoint-2406.predictions.jsonl")
    merged_rows = read_jsonl(root / "tasks/stage2-event-tree/runs/C-42-20260918/checkpoint-eval/dev-checkpoint-2406.merged.jsonl")
    run = json.loads((root / "tasks/stage2-event-tree/runs/C-42-20260918/run.json").read_text(encoding="utf-8"))
    stored_path = root / "tasks/stage2-event-tree/runs/C-42-20260918/checkpoint-eval/dev-checkpoint-2406.metrics.json"
    stored_metrics = json.loads(stored_path.read_text(encoding="utf-8"))
    schemas = read_schema(root / "task/scievents_appendix_f2_exact_reference_v2.md")
    raw_dev = json.loads((root / "data/SciEvents/dev.json").read_text(encoding="utf-8"))
    raw_train = json.loads((root / "data/SciEvents/train.json").read_text(encoding="utf-8"))
    if run.get("status") != "completed" or Path(run["best_checkpoint"]).name != "checkpoint-2406":
        raise ValueError("Original C run/checkpoint state does not match the declared analysis source")
    if len({r["id"] for r in dev_rows}) != len(dev_rows) or len({p["id"] for p in pred_rows}) != len(pred_rows):
        raise ValueError("Duplicate sample IDs")
    pred_by_id = {row["id"]: row for row in pred_rows}
    merged_by_id = {row["id"]: row for row in merged_rows}
    if set(pred_by_id) != {row["id"] for row in dev_rows}:
        raise ValueError("Prediction IDs do not exactly match sentence-C dev IDs")
    if len(merged_by_id) != len(merged_rows) or set(merged_by_id) != {doc["doc_id"] for doc in raw_dev}:
        raise ValueError("Merged prediction document IDs do not match raw dev")
    raw_dev_ids = {doc["doc_id"] for doc in raw_dev}
    if raw_dev_ids != {row["document_id"] for row in dev_rows}:
        raise ValueError("Raw dev document IDs do not match sentence-C dev")
    if len(raw_train) != len({row["document_id"] for row in train_rows}):
        raise ValueError("Train data document counts mismatch")
    if any(pred_by_id[row["id"]].get("invalid") for row in dev_rows):
        raise ValueError("Invalid prediction rows found")

    stats = _role_statistics(train_rows, schemas)
    same_span, nested, shared = _doc_argument_relations(train_rows + dev_rows)
    relations = (same_span, nested, shared)
    event_features, samples = [], []
    confusion = Counter()
    confusion_matches = []
    match_cache = {}
    type_counts = Counter()
    trigger_lexicon = _normalize_trigger_lexicon(train_rows)
    no_event_detail = []
    duplicate_trigger_docs = []
    canonical_gold_by_doc = defaultdict(list)
    canonical_pred_by_doc = defaultdict(list)

    for row in dev_rows:
        prediction = pred_by_id[row["id"]]
        merged_doc = merged_by_id[row["document_id"]]
        lo, hi = row["target_bounds"]
        # Reuse the official merged/aligned event offsets. Re-aligning each sentence
        # independently changes C's document-level argument tie-breaks and deduplication.
        pred_events = [event for event in merged_doc["aligned_events"]
                       if event.get("trigger", {}).get("offsets")
                       and all(lo <= offset < hi for offset in event["trigger"]["offsets"])]
        alignment_stats = {"events": len(pred_events), "aligned_events": len(pred_events)}
        gold_events = row["target_events"]
        matches, missed, spurious = match_events(gold_events, pred_events)
        match_cache[row["id"]] = {"matches": matches, "missed": missed, "spurious": spurious}
        for gi in missed:
            confusion[(gold_events[gi]["event_type"], "MISSED")] += 1
        for pi in spurious:
            confusion[("SPURIOUS", pred_events[pi]["event_type"])] += 1
        for gi, pi, tier in matches:
            gt, pt = gold_events[gi]["event_type"], pred_events[pi]["event_type"]
            confusion[(gt, pt)] += 1
            confusion_matches.append({"sample_id": row["id"], "gold_event_index": gi,
                                      "pred_event_index": pi, "gold_type": gt,
                                      "pred_type": pt, "match_tier": tier,
                                      "type_correct": gt == pt})
        gold_types = [event["event_type"] for event in gold_events]
        pred_types = [event["event_type"] for event in pred_events]
        type_metric = multiset_scores(gold_types, pred_types)
        local_features = [_event_features(row, event, ei, schemas, stats, relations)
                          for ei, event in enumerate(gold_events)]
        event_features.extend(local_features)
        type_counts.update(gold_types)
        gold_trigger_counter = Counter(tuple(sorted(set(event["trigger"]["offsets"]))) for event in gold_events)
        if any(count > 1 for count in gold_trigger_counter.values()):
            duplicate_trigger_docs.append(row["id"])
        sample = {
            "sample_id": row["id"], "abstract_id": row["document_id"],
            "target_sentence_id": row["sentence_index"], "gold_event_count": len(gold_events),
            "pred_event_count": len(pred_events), "is_no_event": len(gold_events) == 0,
            "gold_events": gold_events, "pred_events": pred_events,
            "gold_types": gold_types, "pred_types": pred_types,
            "sample_min_recoverability": min((f["weighted_schema_recoverability"] for f in local_features), default=None),
            "sample_mean_recoverability": (sum(f["weighted_schema_recoverability"] for f in local_features) / len(local_features) if local_features else None),
            "sample_min_schema_margin": min((f["schema_margin"] for f in local_features), default=None),
            "sample_mean_schema_margin": (sum(f["schema_margin"] for f in local_features) / len(local_features) if local_features else None),
            "has_complex_argument": any(f["has_discontinuous"] or f["has_nested"] or f["has_shared"] or f["has_reverse"] or f["has_cross_sentence_argument"] for f in local_features),
            "has_cross_sentence_argument": any(f["has_cross_sentence_argument"] for f in local_features),
            "type_only_exact": type_metric["exact"],
            "pred_invalid": bool(prediction.get("invalid")),
            "alignment": alignment_stats,
            "event_features": local_features,
            "sample_row": row,
        }
        samples.append(sample)
        if sample["is_no_event"]:
            trigger_count = _trigger_like_count(row["target_sentence"], trigger_lexicon)
            no_event_detail.append({"sample_id": row["id"], "trigger_like_match_count": trigger_count,
                                    "trigger_like_bin": "N1:0" if trigger_count == 0 else "N2:1" if trigger_count == 1 else "N3:2+",
                                    "pred_event_count": len(pred_events),
                                    "predicted_types": _csv_list(pred_types),
                                    "false_positive": bool(pred_events)})
        canonical_gold_by_doc[row["document_id"]].extend(gold_events)
        canonical_pred_by_doc[row["document_id"]].extend(pred_events)

    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    # Pooled official metric reproduction; sentence buckets keep the target-sentence ownership explicit.
    full_official = _official_for_samples(samples)
    official_diffs = {key: abs(full_official[key] - stored_metrics["official"][key])
                      for key in METRIC_KEYS}
    if max(official_diffs.values()) > 1e-9:
        raise ValueError(f"Recomputed C metrics do not reproduce stored metrics: {official_diffs}")

    event_count_rows = []
    event_count_groups = {"0": [], "1": [], "2": [], "3+": []}
    for sample in samples:
        count = sample["gold_event_count"]
        label = "3+" if count >= 3 else str(count)
        event_count_groups[label].append(sample)
    for label in ("0", "1", "2", "3+"):
        record = _slice_record(label, event_count_groups[label])
        if label == "0":
            for key in ("type_p", "type_r", "type_f1"):
                record[key] = None
            for key in METRIC_KEYS:
                record[key] = None
        event_count_rows.append(record)

    positive = [sample for sample in samples if sample["gold_event_count"] > 0]
    rec_labels = _quantile_labels(positive, "sample_min_recoverability")
    margin_labels = _quantile_labels(positive, "sample_min_schema_margin")
    for sample in samples:
        sample["recoverability_bucket"] = rec_labels.get(sample["sample_id"], "")
        sample["schema_margin_bucket"] = margin_labels.get(sample["sample_id"], "")
    recoverability_rows, margin_rows = [], []
    for label in ("Low", "Medium", "High"):
        selected = [s for s in positive if s["recoverability_bucket"] == label]
        if selected:
            recoverability_rows.append(_slice_record(label, selected))
        selected = [s for s in positive if s["schema_margin_bucket"] == label]
        if selected:
            margin_rows.append(_slice_record(label, selected))
    interaction_rows = []
    for count_label in ("1", "2", "3+"):
        for margin_label in ("High", "Medium", "Low"):
            selected = [s for s in positive
                        if ("3+" if s["gold_event_count"] >= 3 else str(s["gold_event_count"])) == count_label
                        and s["schema_margin_bucket"] == margin_label]
            interaction_rows.append(_slice_record(f"{count_label} events / {margin_label} margin", selected))

    # Event-level type confusion. Row = gold type; column = predicted type; unmatched endpoints explicit.
    confusion_rows = []
    matrix_columns = list(EVENT_TYPES) + ["MISSED"]
    for gt in EVENT_TYPES:
        confusion_rows.append({"gold_type": gt, **{pt: confusion[(gt, pt)] for pt in matrix_columns}})
    confusion_rows.append({"gold_type": "SPURIOUS", **{pt: confusion[("SPURIOUS", pt)] for pt in EVENT_TYPES}, "MISSED": 0})
    confusion_rows.append({"gold_type": "TOTAL", **{pt: sum(confusion[(gt, pt)] for gt in EVENT_TYPES) + (confusion[("SPURIOUS", pt)] if pt in EVENT_TYPES else 0) for pt in matrix_columns}})
    confusion_rows[-1]["MISSED"] = sum(confusion[(gt, "MISSED")] for gt in EVENT_TYPES)
    type_confusion_stats = {}
    for gt in EVENT_TYPES:
        type_confusion_stats[gt] = {
            "gold": sum(confusion[(gt, pt)] for pt in EVENT_TYPES) + confusion[(gt, "MISSED")],
            "correct": confusion[(gt, gt)],
            "wrong_type": sum(confusion[(gt, pt)] for pt in EVENT_TYPES if pt != gt),
            "missed": confusion[(gt, "MISSED")],
            "spurious_as_this_type": confusion[("SPURIOUS", gt)],
        }
    confusion_pairs = [{"gold_type": gt, "pred_type": pt, "count": confusion[(gt, pt)],
                        "gold_support": type_confusion_stats[gt]["gold"],
                        "rate_given_gold": confusion[(gt, pt)] / type_confusion_stats[gt]["gold"] if type_confusion_stats[gt]["gold"] else 0.0}
                       for gt in EVENT_TYPES for pt in EVENT_TYPES if pt != gt and confusion[(gt, pt)] > 0]
    confusion_pairs.sort(key=lambda row: (-row["count"], row["gold_type"], row["pred_type"]))
    schema_matrix_rows, schema_matrix, schema_pairs = _schema_similarity_matrix(schemas, stats["idf"])
    schema_pair_keys = {(min(row["type_a"], row["type_b"]), max(row["type_a"], row["type_b"])) for row in schema_pairs[:10]}
    top_confusions = confusion_pairs[:10]
    confusion_pair_keys = {(min(r["gold_type"], r["pred_type"]), max(r["gold_type"], r["pred_type"])) for r in top_confusions}
    schema_confusion_intersection = sorted(schema_pair_keys & confusion_pair_keys)
    corr_x, corr_y = [], []
    for gt in EVENT_TYPES:
        for pt in EVENT_TYPES:
            if gt != pt:
                corr_x.append(schema_matrix[gt][pt])
                corr_y.append(type_confusion_stats[gt]["gold"] and confusion[(gt, pt)] / type_confusion_stats[gt]["gold"] or 0.0)
    schema_confusion_spearman = _spearman(corr_x, corr_y)

    # Gold-conditioned complexity slices: each selected gold event is paired only with its event-level match.
    complexity_names = ["normal_contiguous", "same_span_reuse", "nested", "shared",
                        "discontinuous_2frag_gap_le_3", "discontinuous_2frag_gap_4_102",
                        "discontinuous_2frag_gap_gt_102", "discontinuous_3plus", "reverse", "cross_sentence"]
    complexity_rows = []
    for category in complexity_names:
        selected_gold, selected_pred, selected_args = [], [], 0
        selected_events = 0
        for sample in samples:
            cache = match_cache[sample["sample_id"]]
            pred_for_gold = {gi: pi for gi, pi, _ in cache["matches"]}
            for gi, feature in enumerate(sample["event_features"]):
                if category not in _complexity_categories(feature):
                    continue
                selected_events += 1
                selected_args += len(sample["gold_events"][gi]["arguments"])
                selected_gold.append([sample["gold_events"][gi]])
                pi = pred_for_gold.get(gi)
                selected_pred.append([sample["pred_events"][pi]] if pi is not None else [])
        scores = official_scores(selected_gold, selected_pred) if selected_gold else {}
        complexity_rows.append({"complexity": category, "gold_events": selected_events,
                                "gold_arguments": selected_args,
                                "AI_f1": scores.get("AI_f1"), "AC_f1": scores.get("AC_f1"),
                                "matched_event_pairs": sum(1 for sample in samples for gi, _, _ in match_cache[sample["sample_id"]]["matches"]
                                                            if category in _complexity_categories(sample["event_features"][gi]))})

    no_event_rows = []
    no_event_n = len(no_event_detail)
    for label in ("N1:0", "N2:1", "N3:2+"):
        group = [row for row in no_event_detail if row["trigger_like_bin"] == label]
        predicted_types = Counter(t for row in group for t in row["predicted_types"].split("|") if t)
        event_count_distribution = Counter(str(row["pred_event_count"]) for row in group)
        no_event_rows.append({"trigger_like_bin": label, "samples": len(group),
                              "false_positive_samples": sum(row["false_positive"] for row in group),
                              "false_positive_rate": sum(row["false_positive"] for row in group) / len(group) if group else None,
                              "predicted_events": sum(row["pred_event_count"] for row in group),
                              "predicted_event_count_distribution": json.dumps(dict(sorted(event_count_distribution.items())), ensure_ascii=False),
                              "predicted_type_counts": json.dumps(dict(predicted_types), ensure_ascii=False, sort_keys=True)})

    # Write requested row-level feature tables.
    feature_fields = ["sample_id", "abstract_id", "target_sentence_id", "gold_event_index", "gold_event_type",
                      "trigger_text", "num_arguments", "observed_roles", "missing_roles", "plain_role_coverage",
                      "weighted_schema_recoverability", "gold_schema_score", "best_competing_type",
                      "best_competing_score", "schema_margin", "has_cross_sentence_argument",
                      "has_same_span_reuse", "has_nested", "has_shared", "has_discontinuous", "has_reverse",
                      "max_num_fragments", "max_gap_tokens"]
    write_csv(output_dir / "event_level_features.csv", event_features, feature_fields)
    sample_out = [{key: sample[key] for key in (
        "sample_id", "abstract_id", "target_sentence_id", "gold_event_count", "pred_event_count", "is_no_event",
        "sample_min_recoverability", "sample_mean_recoverability", "sample_min_schema_margin",
        "sample_mean_schema_margin", "has_complex_argument", "has_cross_sentence_argument", "type_only_exact",
        "recoverability_bucket", "schema_margin_bucket")}
                  for sample in samples]
    write_csv(output_dir / "sample_level_features.csv", sample_out)
    write_csv(output_dir / "schema_similarity.csv", schema_matrix_rows)
    write_csv(output_dir / "event_type_confusion.csv", confusion_rows)
    write_csv(output_dir / "schema_similarity_pairs.csv", schema_pairs,
              ["type_a", "type_b", "weighted_similarity", "unweighted_jaccard"])
    write_csv(output_dir / "difficulty_slices.csv", event_count_rows + recoverability_rows + margin_rows + interaction_rows)
    write_csv(output_dir / "argument_complexity.csv", complexity_rows)
    write_csv(output_dir / "no_event_breakdown.csv", no_event_rows)
    write_csv(output_dir / "event_alignment.csv", confusion_matches,
              ["sample_id", "gold_event_index", "pred_event_index", "gold_type", "pred_type", "match_tier", "type_correct"])
    write_csv(output_dir / "no_event_examples.csv", no_event_detail)

    overall_type = _type_only_for_samples(samples)
    count_match = sum(sample["gold_event_count"] == sample["pred_event_count"] for sample in samples)
    matched_event_count = sum(len(match_cache[s["sample_id"]]["matches"]) for s in samples)
    spurious_event_count = sum(len(match_cache[s["sample_id"]]["spurious"]) for s in samples)
    gold_event_count_from_matrix = sum(type_confusion_stats[t]["gold"] for t in EVENT_TYPES)
    if gold_event_count_from_matrix != sum(len(s["gold_events"]) for s in samples):
        raise ValueError("Gold event confusion accounting does not reconcile")
    if matched_event_count + spurious_event_count != sum(len(s["pred_events"]) for s in samples):
        raise ValueError("Predicted event confusion accounting does not reconcile")
    count_rows = {row["slice"]: row for row in event_count_rows}
    margin_count_rows = {}
    for row in interaction_rows:
        count_part, margin_part = row["slice"].split(" / ")
        margin_count_rows[(count_part.replace(" events", ""), margin_part.replace(" margin", ""))] = row
    rec_values = [row["type_f1"] for row in recoverability_rows]
    rec_exact = [row["exact"] for row in recoverability_rows]
    margin_values = [row["type_f1"] for row in margin_rows]
    margin_exact = [row["exact"] for row in margin_rows]
    count_exact_by_margin = {label: [margin_count_rows.get((count, label), {}).get("exact")
                                     for count in ("1", "2", "3+")] for label in ("High", "Medium", "Low")}
    summary = {
        "source": {
            "train_sentence_rows": len(train_rows), "dev_sentence_rows": len(dev_rows),
            "dev_prediction_rows": len(pred_rows), "dev_documents": len(raw_dev),
            "gold_events": sum(len(s["gold_events"]) for s in samples),
            "predicted_events": sum(len(s["pred_events"]) for s in samples),
            "invalid_predictions": sum(s["pred_invalid"] for s in samples),
            "best_checkpoint": Path(run["best_checkpoint"]).name,
            "duplicate_gold_trigger_samples": duplicate_trigger_docs,
        },
        "official_dev": full_official, "stored_official_dev": stored_metrics["official"],
        "official_metric_abs_diff": official_diffs,
        "type_only": {**overall_type, "exact_count_only_accuracy": count_match / len(samples) if samples else None},
        "event_count_slices": event_count_rows, "recoverability_slices": recoverability_rows,
        "schema_margin_slices": margin_rows, "event_count_x_margin": interaction_rows,
        "type_confusion_by_gold": type_confusion_stats, "top_type_confusions": top_confusions,
        "top_schema_similarity_pairs": schema_pairs[:10],
        "top_schema_confusion_pair_intersection": schema_confusion_intersection,
        "schema_similarity_vs_confusion_spearman_n90": schema_confusion_spearman,
        "argument_complexity": complexity_rows, "no_event_breakdown": no_event_rows,
        "no_event_samples": no_event_n,
        "event_count_exact_by_margin": count_exact_by_margin,
        "curriculum_signals": {
            "recoverability_type_f1_monotonic_low_to_high": all(a <= b for a, b in zip(rec_values, rec_values[1:])),
            "recoverability_exact_monotonic_low_to_high": all(a <= b for a, b in zip(rec_exact, rec_exact[1:])),
            "margin_type_f1_monotonic_low_to_high": all(a <= b for a, b in zip(margin_values, margin_values[1:])),
            "margin_exact_monotonic_low_to_high": all(a <= b for a, b in zip(margin_exact, margin_exact[1:])),
            "count_exact_monotonic_within_margin": {label: all(v1 is not None and v2 is not None and v1 >= v2
                for v1, v2 in zip(values, values[1:])) for label, values in count_exact_by_margin.items()},
        },
        "role_stats_train_only": {event_type: {"events": stats["totals"][event_type],
                                                "p_role": stats["p_role"][event_type]}
                                   for event_type in EVENT_TYPES},
        "definitions": {
            "schema_margin": "weighted Jaccard of observed role set against gold schema minus best other schema; role weights are train-derived IDF",
            "event_matching": "one-to-one, type-agnostic maximum-weight assignment; trigger exact > overlap > normalized text > argument span/role evidence",
            "nested": "strict token-set containment between distinct argument spans in the same abstract; equal spans excluded",
            "shared": "same exact sorted token-offset set used by arguments of at least two distinct gold events in the same abstract",
            "same_span_reuse": "same exact token-offset span repeated within one event",
            "reverse": "all argument tokens occur before all trigger tokens",
            "normal_contiguous": "event has at least one argument and every argument offset span is contiguous",
            "gap_tokens": "number of unannotated document tokens between adjacent contiguous fragments",
            "trigger_lexicon": "distinct train-gold trigger token sequences; exact casefolded contiguous match in target sentence",
        },
    }
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / "difficulty_analysis_report.md", summary, schemas)
    (output_dir / "verification.json").write_text(json.dumps({
        "passed": True, "source_counts": summary["source"],
        "official_metric_abs_diff": official_diffs,
        "gold_accounting": {"confusion_row_total": gold_event_count_from_matrix,
                            "gold_events": summary["source"]["gold_events"]},
        "prediction_accounting": {"matched": matched_event_count, "spurious": spurious_event_count,
                                  "matched_plus_spurious": matched_event_count + spurious_event_count,
                                  "predicted_events": summary["source"]["predicted_events"]},
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "verification.json"},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_report(path, summary, schemas):
    def pct(value):
        return "NA" if value is None else f"{value * 100:.2f}%"
    lines = ["# SciEvents 原始 C 树：Curriculum Difficulty Analysis", "",
             "## 数据与口径", "",
             f"使用原始 C 的 `{summary['source']['best_checkpoint']}` dev-best 预测；未训练模型、未重新推理。分析 {summary['source']['dev_sentence_rows']:,} 个目标句，覆盖 {summary['source']['dev_documents']} 篇摘要；gold/pred 分别有 {summary['source']['gold_events']:,}/{summary['source']['predicted_events']:,} 个事件。无效预测 {summary['source']['invalid_predictions']}。", "",
             "官方 TI/TC/AI/AC 通过项目 scorer 对每个目标句的完整事件列表打分后 pooled 汇总；argument offsets 仍保留文档级位置。type-only 是每句事件类型多重集合，不看 trigger。事件类型混淆用 type-agnostic 一对一对齐，依次使用 trigger exact、trigger overlap、normalized trigger text、argument span/role overlap；事件类型不参与配对。", "",
             "**Metric verification:** full-dev official score is compared with the stored checkpoint metrics; absolute differences: " + ", ".join(f"{k}={v:.2g}" for k,v in summary["official_metric_abs_diff"].items()) + ".", "",
             "## 总体表现", "",
             f"Type-only P/R/F1 = **{pct(summary['type_only']['type_p'])}/{pct(summary['type_only']['type_r'])}/{pct(summary['type_only']['type_f1'])}**；整句 type inventory exact = **{pct(summary['type_only']['exact'])}** ({summary['type_only']['exact'] * summary['source']['dev_sentence_rows']:.0f}/{summary['source']['dev_sentence_rows']})。只看事件数量、忽略类型的句子准确率为 {pct(summary['type_only']['exact_count_only_accuracy'])}。", "",
             "官方 pooled 指标：" + ", ".join(f"{k}={pct(v)}" for k,v in summary["official_dev"].items()), "",
             "## Q1–Q3：schema difficulty 与事件数量", "",
             "### Event count slices", "",
             _report_table(summary["event_count_slices"], ["slice", "samples", "exact", "type_p", "type_r", "type_f1", "TI_f1", "TC_f1", "AI_f1", "AC_f1"]), "",
             "### Schema recoverability（positive samples，按 sample-min train-weighted recoverability 分位分桶）", "",
             _report_table(summary["recoverability_slices"], ["slice", "samples", "exact", "type_f1", "TI_f1", "TC_f1", "AI_f1", "AC_f1"]), "",
             "### Schema margin（positive samples，按 sample-min margin 分位分桶）", "",
             _report_table(summary["schema_margin_slices"], ["slice", "samples", "exact", "type_f1", "TI_f1", "TC_f1", "AI_f1", "AC_f1"]), "",
             "### Event count × schema margin（margin 由高到低；type inventory exact 为主）", "",
             _report_table(summary["event_count_x_margin"], ["slice", "samples", "exact", "type_f1", "TC_f1", "AC_f1"]), "",
             "判断：比较表中同 event-count 行和同 margin 列的变化，不预设单调关系。分桶按样本-min 分数的经验三分位，边界相同的 tie 留在同一组，故各组样本数不保证完全相等。", "",
             "## Q4–Q5：事件类型错误", "",
             "混淆配对表中，wrong-type 是匹配到同一个 trigger/argument 证据但类型不同；MISSED 是无可配对预测；SPURIOUS 是无可配对 gold。该分析和 type-only multiset 分数互补。", "",
             "| Gold type | gold n | correct | wrong-type | missed | spurious as type |", "|---|---:|---:|---:|---:|---:|"]
    for event_type in EVENT_TYPES:
        row = summary["type_confusion_by_gold"][event_type]
        lines.append(f"| {event_type} | {row['gold']} | {row['correct']} | {row['wrong_type']} | {row['missed']} | {row['spurious_as_this_type']} |")
    lines.extend(["", "Top confusion pairs：", "", "| Gold → Pred | Count | Rate among gold type |", "|---|---:|---:|"])
    for row in summary["top_type_confusions"]:
        lines.append(f"| {row['gold_type']} → {row['pred_type']} | {row['count']} | {pct(row['rate_given_gold'])} |")
    lines.extend(["", "Top Appendix schema similarity pairs：", "", "| Type pair | Weighted Jaccard | Unweighted Jaccard | Also top-10 confusion pair? |", "|---|---:|---:|---|"])
    top_conf_set = {(min(r["gold_type"], r["pred_type"]), max(r["gold_type"], r["pred_type"])) for r in summary["top_type_confusions"]}
    for row in summary["top_schema_similarity_pairs"]:
        pair = (row["type_a"], row["type_b"])
        lines.append(f"| {pair[0]} ↔ {pair[1]} | {row['weighted_similarity']:.3f} | {row['unweighted_jaccard']:.3f} | {'yes' if pair in top_conf_set else 'no'} |")
    lines.extend(["", f"Spearman ρ between directed off-diagonal schema similarity and gold-normalized model confusion: {summary['schema_similarity_vs_confusion_spearman_n90']!s} (n=90 directed pairs). This is descriptive, not causal.", "",
                  "## Q6：论元结构复杂度", "",
                  "按 gold event 的几何特征分片；这些分片允许重叠。AI/AC 是对每个结构 gold event 与其一对一匹配预测单独调用官方 scorer 后 pooled 的分数，missed gold 保留为空预测；未匹配 spurious prediction 不归入 gold 复杂度切片。", "",
                  _report_table(summary["argument_complexity"], ["complexity", "gold_events", "gold_arguments", "matched_event_pairs", "AI_f1", "AC_f1"]), "",
                  "定义：nested 为同摘要内不同论元 span 的严格 token-offset 包含关系（相等 span 排除）；shared 为至少两个不同事件使用完全相同 offset span；same-span reuse 是同一事件内部重复 span；reverse 指论元整体位于 trigger 前；cross-sentence 指至少一个论元 token 不在 trigger 所属句。句法距离等不能从这些标注字段推断。", "",
                  "## Q7：No-event false positives", "",
                  f"Gold no-event 样本数：{summary['no_event_samples']}。trigger-like lexicon 仅从 train gold 的 trigger token sequence 建立，按目标句内大小写归一后的连续精确匹配计数。", "",
                  _report_table(summary["no_event_breakdown"], ["trigger_like_bin", "samples", "false_positive_samples", "false_positive_rate", "predicted_events", "predicted_event_count_distribution", "predicted_type_counts"]), "",
                  "该 lexicon 只表示训练中出现过同一 trigger surface，不代表句中真有事件；因此分层结果是触发词表命中与误报的关联诊断。", "",
                  "## Q8：Curriculum 结论", "",
                  "结果解释（与单一的逐级难度假设区分）：recoverability 从 Low→Medium→High 时，type-only F1 为 71.83→76.08→80.49%，句级 type inventory exact 为 40.72→50.00→64.90%，呈单调上升，因此它是本次数据中最清楚的 curriculum 维度。Schema margin 的 exact 为 Low 36.40%、Medium 60.37%、High 58.12%；Low 明显最难，但 Medium/High 不单调，不能宣称完整线性阶梯。", "",
                  "事件数主要降低整句 inventory exact：1/2/3+ 事件分别为 61.82/50.30/29.60%；但对应 type-only F1 是 74.72/77.37/73.79%，并不随事件数单调下降。固定 margin 后 exact 在各档都随事件数增加而下降；margin 在固定事件数时 Low 也始终落后，2-event 与 3+-event 的 High→Medium→Low 呈单调下降，单事件组则 Medium(67.65%) 高于 High(62.44%)，Low(45.10%) 最差。故建议把“多事件组合”和“低 margin”作为两种独立难度轴。", "",
                  "类型错误的例子：WKS→MDS 有 52 个、MDS→PUR 36 个、MDS→WKS 32 个、PUR→MDS 27 个、CMP→FAC 25 个。Appendix schema 中 PRP↔WKS 同时是 top-10 相似 pair 和 top-10 confusion pair；整体 schema similarity 与 confusion 的 Spearman ρ=0.256（90 个有向类型对），关联偏弱，schema 重叠只能解释部分混淆。RWS 的 70 个 gold 中 41 个漏检、12 个错分；WKS 则 82 个错分、73 个漏检，故不同类型的主要错误机制不同。", "",
                  "结构复杂度的结果支持把连续论元放在前、结构困难样本逐步混入：普通连续组 n=2,257，AI/AC=46.63/43.95%；跨句组 n=282 为 38.67/37.36%；嵌套组 n=166 为 39.04/37.26%；两片段且 gap 4–102 的组 n=55 为 28.26/26.81%；三片段以上 n=33 为 20.98/20.98%。但 shared 组 n=678 的 AI/AC=49.08/47.49%，反而较高；reverse n=1,526 也没有明显变差，因此不应把所有非标准结构都视作同等困难。gap>102 仅 n=11，只作提示。", "",
                  "No-event 结果：271 个 gold-empty 句中有 60 个误报（22.14%），全部发生在 train trigger lexicon 命中至少 2 次的组（60/259=23.17%，对应 92 个预测事件）；0/1 次命中组是 0/3 与 0/9，样本太少不能据此断言安全。课程上可先用高 recoverability、单事件样本，再增加高/中 margin 多事件组合，之后加低 margin 与跨句、嵌套、断续论元；shared/reverse 混合贯穿各阶段，并在最终阶段混入全部分布。该顺序是基于切片的设计建议，不是已验证的训练收益；要证明收益仍需做受控训练实验。", "",
                  "## 可复现产物", "",
                  "- `analyze_difficulty.py`：全流程脚本；`python3 tasks/stage2-event-tree/difficulty-analysis/analyze_difficulty.py` 重跑。",
                  "- `event_level_features.csv`、`sample_level_features.csv`：gold difficulty 特征。",
                  "- `schema_similarity.csv`、`event_type_confusion.csv`：schema 和模型 type diagnostics。",
                  "- `difficulty_slices.csv`、`argument_complexity.csv`、`no_event_breakdown.csv`：分层结果。",
                  "- `event_alignment.csv`：type-agnostic event pairs 和匹配层级。",
                  "- `verification.json`：输入计数、官方指标复核及输出文件 hash。",
                  "", "### Limitations", "",
                  "Event-type confusion depends on event alignment. Exact trigger duplicates are ambiguous in the official offset scorer (its span dictionaries retain one event per identical trigger key); duplicate-trigger sample IDs are listed in `analysis_summary.json`. Prediction argument offsets are reconstructed from output text by the existing aligner and may be ambiguous. Gold structural properties use raw gold offsets. Semantic coreference and discourse-level difficulty are not recoverable from current spans."])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tasks/stage2-event-tree/difficulty-analysis")
    args = parser.parse_args()
    summary = analyze(args.project_root, args.output_dir)
    print(json.dumps({"source": summary["source"], "official_dev": summary["official_dev"],
                      "type_only": summary["type_only"], "top_type_confusions": summary["top_type_confusions"][:5]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
