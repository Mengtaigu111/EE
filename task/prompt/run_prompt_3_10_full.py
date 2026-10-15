"""Checkpointed, two-way batch generation for the train split with a Type-CoT prompt."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent
ROOT = PROMPT_DIR.parents[1]
DATA_DIR = ROOT / "data" / "SciEvents"
PROMPT_PATH = Path(os.environ.get("PROMPT_FILE", str(PROMPT_DIR / "3.10.md")))
REF_PATH = ROOT / "task" / "scievents_appendix_f2_exact_reference_v2.md"
API_URL = os.environ.get("API_URL", "https://api.shuaiapi.com/v1/chat/completions")
PROXY_URL = "http://10.70.115.48:7897"
MODEL = "gpt-5.5"
NO_INVENTORY_OUTPUT = os.environ.get("NO_INVENTORY_OUTPUT", "0") == "1"
ALLOWED_TYPES = ("PUR", "ITT", "RWS", "RWF", "PRP", "WKS", "MDS", "FIN", "CMP", "FAC")
TYPE_RE = "|".join(ALLOWED_TYPES)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_schema():
    text = REF_PATH.read_text(encoding="utf-8")
    schema = {}
    order = list(ALLOWED_TYPES)
    for code in order:
        sec = re.search(rf"^### {code} — (\w+)\n(.*?)(?=^### |^---)", text, re.S | re.M)
        name, body = sec.group(1), sec.group(2)
        blocks = re.findall(r"```text\n(.*?)\n```", body, re.S)
        template = re.sub(r":arg\d+", "", blocks[0].strip())
        roles = blocks[1].split()
        schema[code] = {"name": name, "template": template, "roles": roles}
    return schema


def render_schemas(schema):
    return "\n\n".join(
        f"{code} ({value['name']})\nTemplate: {value['template']}\nRoles: {', '.join(value['roles'])}"
        for code, value in schema.items()
    )


def events_by_sentence(doc):
    bounds, start = [], 0
    for sentence in doc["sentences"]:
        end = start + len(sentence.split())
        bounds.append((start, end))
        start = end
    output = {}
    for event in doc["events"]:
        offset = event["trigger"]["offsets"][0]
        sentence_index = next(i for i, (left, right) in enumerate(bounds) if left <= offset < right)
        output.setdefault(sentence_index, []).append(event)
    for events in output.values():
        events.sort(key=lambda event: event["trigger"]["offsets"][0])
    return output


def render_event(index, event, schema):
    current = schema[event["event_type"]]
    by_role = {}
    for argument in event["arguments"]:
        by_role.setdefault(argument["argument_type"], []).append(argument["text"])

    def fill(match):
        role = match.group(1)
        return " and ".join(f"[{role}: {text}]" for text in by_role.get(role, ["unspecified"]))

    filled = re.sub(r"<(\w+)>", fill, current["template"])
    args = "\n".join(f"- {arg['argument_type']}: {arg['text']}" for arg in event["arguments"]) or "- (none)"
    return f"Event {index}:\nEvent Type: {event['event_type']}\nFilled Template: {filled}\nArguments:\n{args}"


def render_gold_events(events, schema):
    if not events:
        return "None"
    return "\n\n".join(render_event(i, event, schema) for i, event in enumerate(events, 1))


def inventory(events):
    counts = Counter(event["event_type"] for event in events)
    return "\n".join(f"{code}: {counts[code]}" for code in ALLOWED_TYPES if counts[code]) or "None"


def read_key(config_path, index=0):
    keys = []
    for line in Path(config_path).read_text(encoding="utf-8").splitlines():
        match = re.search(r"\b(sk-[A-Za-z0-9_-]+)\b", line)
        if match:
            keys.append(match.group(1))
    if index >= len(keys):
        raise RuntimeError(f"API key index {index} not found in {config_path}")
    return keys[index]


def fixed_none(system_prompt):
    if NO_INVENTORY_OUTPUT:
        return (
            "Reasoning:\n\n"
            "1. Event Family Reasoning:\n"
            "   The target sentence does not express a semantic relation corresponding to any event type defined by the schemas, so no event-specific reasoning entry is generated."
        )
    marker = "If the event type inventory is None, return exactly:"
    if marker in system_prompt:
        tail = system_prompt.split(marker, 1)[1].strip()
        fenced = re.search(r"```text\n(.*?)\n```", tail, re.S)
        if fenced:
            return fenced.group(1)
        return tail
    raise RuntimeError("prompt 3.10 fixed None completion was not found")


def build_user(sample, events, schema):
    return (
        "[Abstract]\n"
        + sample["abstract"]
        + "\n\n[Target Sentence]\n"
        + sample["target_sentence"]
        + "\n\n[Gold Event Type Inventory]\n"
        + inventory(events)
        + "\n\n[Gold Arguments by Event]\n"
        + render_gold_events(events, schema)
        + "\n\n[Official Event Schemas]\n"
        + render_schemas(schema)
        + "\n\nGenerate the reasoning chain following the required procedure."
    )


def call_api(system_prompt, user_prompt, key):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4000,
        }
    ).encode("utf-8")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY_URL, "https": PROXY_URL})
    )
    last_error = "unknown error"
    for attempt in range(1, 5):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener.open(request, timeout=240) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = type(exc).__name__
        if attempt < 4:
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(last_error)


def parse_inventory(text):
    normalized = re.sub(rf"<?({TYPE_RE})(?:></type>|</type>)", r"\1", text)
    normalized = re.sub(r"<?(\d+)(?:></count>|</count>)", r"\1", normalized)
    if "Event Type Inventory:" not in normalized:
        return None
    tail = normalized.split("Event Type Inventory:", 1)[1]
    if re.search(r"\bNone\b", tail) and not re.search(rf"\b(?:{TYPE_RE})\s*:\s*\d+", tail):
        return Counter()
    result = Counter()
    for code, count in re.findall(rf"\b({TYPE_RE})\s*:\s*(\d+)\b", tail):
        result[code] += int(count)
    return result


def qc_output(text, events, system_prompt):
    expected_counts = Counter(event["event_type"] for event in events)
    if not events:
        fixed = fixed_none(system_prompt)
        if NO_INVENTORY_OUTPUT:
            return {
                "ok": text == fixed and "Event Type Inventory:" not in text,
                "event_entry_count_ok": True,
                "type_format_ok": True,
                "inventory_output_ok": "Event Type Inventory:" not in text,
                "unknown_types": [],
                "none_fixed_ok": text == fixed,
            }
        return {
            "ok": text == fixed,
            "event_entry_count_ok": True,
            "inventory_count_ok": text == fixed,
            "type_format_ok": True,
            "unknown_types": [],
            "inventory_header_ok": text.count("Event Type Inventory:") == 1,
            "none_fixed_ok": text == fixed,
        }

    blocks = re.findall(
        r"(?ms)^\s*Event \d+:.*?(?=^\s*Event \d+:|^\s*Event Type Inventory:|\Z)", text
    )
    entry_types = []
    type_format_ok = True
    unknown = []
    for block in blocks:
        strict_matches = re.findall(r"Type:\s*<([^>\n]+)></type>", block)
        matches = re.findall(r"Type:\s*<([^>\n]+)></type>?", block)
        if len(matches) != 1:
            type_format_ok = False
            plain = re.findall(r"Type:\s*([^\s\n]+)", block)
            unknown.extend(plain or ["missing"])
        else:
            entry_types.append(matches[0])
            if len(strict_matches) != 1:
                type_format_ok = False
            if matches[0] not in ALLOWED_TYPES:
                unknown.append(matches[0])

    if NO_INVENTORY_OUTPUT:
        return {
            "ok": (
                len(blocks) == len(events)
                and Counter(entry_types) == expected_counts
                and not unknown
                and "Event Type Inventory:" not in text
                and text.rstrip().endswith(("</type>", "</type"))
            ),
            "event_entry_count_ok": len(blocks) == len(events),
            "type_format_ok": type_format_ok,
            "format_warning": not type_format_ok,
            "inventory_output_ok": "Event Type Inventory:" not in text,
            "unknown_types": sorted(set(unknown)),
            "entry_types": entry_types,
            "expected_entry_types": [event["event_type"] for event in events],
        }

    header_ok = bool(re.fullmatch(r".*Event Type Inventory:.*", text, re.S)) and text.count("Event Type Inventory:") == 1 and bool(re.search(r"(?m)^Event Type Inventory:$", text))
    tail = text.split("Event Type Inventory:", 1)[1].strip() if "Event Type Inventory:" in text else ""
    lines = tail.splitlines() if tail else []
    inventory_counts = Counter()
    inventory_format_ok = True
    for line in lines:
        match = re.fullmatch(rf"<({TYPE_RE})></type>:\s*<(\d+)></count>", line.strip())
        if not match:
            inventory_format_ok = False
            unknown_match = re.match(r"<?([^\s>:]+)", line.strip())
            if unknown_match:
                unknown.append(unknown_match.group(1))
        else:
            code, count = match.groups()
            inventory_counts[code] += int(count)
            if code not in ALLOWED_TYPES:
                unknown.append(code)

    normalized = parse_inventory(text) or Counter()
    return {
        "ok": (
            len(blocks) == len(events)
            and Counter(entry_types) == expected_counts
            and normalized == expected_counts
            and type_format_ok
            and inventory_format_ok
            and header_ok
            and not unknown
            and text.rstrip().endswith("</count>")
        ),
        "event_entry_count_ok": len(blocks) == len(events),
        "inventory_count_ok": normalized == expected_counts,
        "type_format_ok": type_format_ok,
        "inventory_format_ok": inventory_format_ok,
        "inventory_header_ok": header_ok,
        "unknown_types": sorted(set(unknown)),
        "entry_types": entry_types,
        "expected_entry_types": [event["event_type"] for event in events],
        "normalized_inventory": dict(normalized),
    }


def sample_records(split="train"):
    docs = load_json(DATA_DIR / f"{split}.json")
    for doc in docs:
        by_sentence = events_by_sentence(doc)
        for sentence_index, sentence in enumerate(doc["sentences"]):
            events = by_sentence.get(sentence_index, [])
            yield {
                "sample_id": f"{split}:{doc['doc_id']}:{sentence_index}",
                "split": split,
                "doc_id": doc["doc_id"],
                "sentence_index": sentence_index,
                "abstract": doc["abstract"],
                "target_sentence": sentence,
                "events": events,
            }


def read_completed(path):
    latest = {}
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[record["sample_id"]] = record
    return {
        sample_id
        for sample_id, record in latest.items()
        if record.get("completed")
        and (record.get("api_used") is False or record.get("qc", {}).get("ok") is True)
    }


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args):
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = load_schema()
    all_samples = list(sample_records("train"))
    selected = all_samples[args.part :: args.parts]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    completed_ids = read_completed(records_path)
    pending = [sample for sample in selected if sample["sample_id"] not in completed_ids]
    expected = len(selected)
    expected_none = sum(not sample["events"] for sample in selected)
    expected_api = expected - expected_none
    manifest = {
        "run": run_dir.name,
        "prompt": str(PROMPT_PATH),
        "split": "train",
        "part": args.part,
        "parts": args.parts,
        "workers": args.workers,
        "model": MODEL,
        "api_key_index": args.api_key_index,
        "sample_attempts": args.sample_attempts,
        "expected_records": expected,
        "expected_none": expected_none,
        "expected_api_calls": expected_api,
        "started_at": now(),
    }
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({**manifest, "already_completed": len(completed_ids), "pending": len(pending)}, ensure_ascii=False), flush=True)
        return 0

    key = read_key(args.api_config, args.api_key_index)
    lock = threading.Lock()
    counts = Counter(completed=len(completed_ids), none=0, api=0, errors=0, qc_pass=0, qc_fail=0)
    if completed_ids:
        latest_records = {}
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest_records[record["sample_id"]] = record
        for record in latest_records.values():
            if not record.get("completed"):
                continue
            if record.get("api_used") is False:
                counts["none"] += 1
            else:
                counts["api"] += 1
            if record.get("qc", {}).get("ok"):
                counts["qc_pass"] += 1
            elif record.get("api_used") is not False:
                counts["qc_fail"] += 1

    def process(sample):
        events = sample["events"]
        gold_inventory = inventory(events)
        retry_history = []
        last_completion = ""
        last_qc = None
        last_error = None
        for attempt in range(1, args.sample_attempts + 1):
            try:
                if not events:
                    completion = fixed_none(system_prompt)
                    api_used = False
                else:
                    completion = call_api(system_prompt, build_user(sample, events, schema), key)
                    api_used = True
                qc = qc_output(completion, events, system_prompt)
                last_completion = completion
                last_qc = qc
                if qc.get("ok"):
                    return {
                        "sample_id": sample["sample_id"],
                        "split": sample["split"],
                        "doc_id": sample["doc_id"],
                        "sentence_index": sample["sentence_index"],
                        "abstract": sample["abstract"],
                        "target_sentence": sample["target_sentence"],
                        "gold_event_type_inventory": gold_inventory,
                        "gold_arguments_by_event": render_gold_events(events, schema),
                        "completion": completion,
                        "api_used": api_used,
                        "completed": True,
                        "qc": qc,
                        "attempts": attempt,
                        "retry_history": retry_history,
                        "created_at": now(),
                    }
                retry_history.append({
                    "attempt": attempt,
                    "kind": "qc_failure",
                    "qc": qc,
                    "completion": completion,
                })
                last_error = "QC failure"
            except Exception as exc:
                last_error = type(exc).__name__ + ": " + str(exc)[:200]
                retry_history.append({"attempt": attempt, "kind": "api_error", "error": last_error})
            if attempt < args.sample_attempts:
                time.sleep(5 * attempt)

        return {
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "doc_id": sample["doc_id"],
            "sentence_index": sample["sentence_index"],
            "gold_event_type_inventory": gold_inventory,
            "completion": last_completion,
            "api_used": bool(events),
            "completed": False,
            "qc": last_qc,
            "attempts": args.sample_attempts,
            "retry_history": retry_history,
            "error": last_error or "retry exhausted",
            "created_at": now(),
        }

    def save_status():
        write_json(
            status_path,
            {
                **manifest,
                "updated_at": now(),
                "counts": dict(counts),
                "pending_at_start": len(pending),
            },
        )

    save_status()
    with records_path.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            batch_size = max(args.workers * 4, 16)
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                futures = [executor.submit(process, sample) for sample in batch]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    with lock:
                        if record["completed"]:
                            counts["completed"] += 1
                            if record.get("api_used"):
                                counts["api"] += 1
                            else:
                                counts["none"] += 1
                            if record.get("qc", {}).get("ok"):
                                counts["qc_pass"] += 1
                            else:
                                counts["qc_fail"] += 1
                        elif record.get("qc"):
                            counts["qc_fail"] += 1
                        else:
                            counts["errors"] += 1
                        done = counts["completed"]
                    if done % 50 == 0 or not record["completed"]:
                        save_status()
                        print(
                            f"part={args.part + 1}/{args.parts} completed={done}/{expected} "
                            f"api={counts['api']} none={counts['none']} errors={counts['errors']} "
                            f"qc_pass={counts['qc_pass']} qc_fail={counts['qc_fail']}",
                            flush=True,
                        )
    save_status()
    summary = {**manifest, "finished_at": now(), "counts": dict(counts), "pending_at_start": len(pending)}
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if counts["errors"] == 0 else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, choices=[0, 1])
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--api-config", required=True)
    parser.add_argument("--api-key-index", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.part >= args.parts:
        raise SystemExit("part must be smaller than parts")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
