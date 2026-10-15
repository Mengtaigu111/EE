"""Render {SCHEMAS} and {GOLD_EVENTS} for task/prompt/3.4.md.

Templates and roles are parsed from task/scievents_appendix_f2_exact_reference_v2.md
(the single source); nothing about the schema is hard-coded here.

Usage:
  python render_gold_events.py --split train --doc 0 --sent 5
  python render_gold_events.py --schemas
  python render_gold_events.py --check          # role coverage over train/dev/test
"""
import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "task" / "scievents_appendix_f2_exact_reference_v2.md"
DATA = ROOT / "data" / "SciEvents"
ORDER = ["PUR", "ITT", "RWS", "RWF", "PRP", "WKS", "MDS", "FIN", "CMP", "FAC"]


def load_schema():
    text = REF.read_text(encoding="utf-8")
    schema = {}
    for code in ORDER:
        sec = re.search(rf"^### {code} — (\w+)\n(.*?)(?=^### |^---)", text, re.S | re.M)
        name, body = sec.group(1), sec.group(2)
        blocks = re.findall(r"```text\n(.*?)\n```", body, re.S)
        template = re.sub(r":arg\d+", "", blocks[0].strip())
        roles = blocks[1].split()
        assert set(re.findall(r"<(\w+)>", template)) == set(roles), code
        schema[code] = {"name": name, "template": template, "roles": roles}
    return schema


def render_schemas(schema):
    return "\n\n".join(
        f"{c} ({s['name']})\nTemplate: {s['template']}\nRoles: {', '.join(s['roles'])}"
        for c, s in schema.items()
    )


def load_split(split):
    text = (DATA / f"{split}.json").read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def events_by_sentence(doc):
    bounds, start = [], 0
    for sent in doc["sentences"]:
        n = len(sent.split())
        bounds.append((start, start + n))
        start += n
    assert start == len(doc["document"]), doc["doc_id"]
    out = collections.defaultdict(list)
    for ev in doc["events"]:
        off = ev["trigger"]["offsets"][0]
        out[next(i for i, (a, b) in enumerate(bounds) if a <= off < b)].append(ev)
    for evs in out.values():
        evs.sort(key=lambda e: e["trigger"]["offsets"][0])  # order only; the trigger itself is never rendered
    return out


def render_event(k, ev, schema):
    s = schema[ev["event_type"]]
    by_role = collections.defaultdict(list)
    for arg in ev["arguments"]:
        by_role[arg["argument_type"]].append(arg["text"])

    def fill(m):
        role = m.group(1)
        return " and ".join(f"[{role}: {t}]" for t in by_role.get(role, ["unspecified"]))

    args = "\n".join(f"- {a['argument_type']}: {a['text']}" for a in ev["arguments"]) or "- (none)"
    filled = re.sub(r"<(\w+)>", fill, s["template"])
    return (
        f"Event {k}:\n"
        f"Event Type: {ev['event_type']}\n"
        f"Filled Template: {filled}\n"
        f"Arguments:\n{args}"
    )


def render_gold_events(evs, schema):
    if not evs:
        return "None"
    return "\n\n".join(render_event(k, ev, schema) for k, ev in enumerate(evs, 1))


def inventory(evs):
    c = collections.Counter(e["event_type"] for e in evs)
    return "\n".join(f"{t}: {c[t]}" for t in ORDER if c[t]) or "None"


def check(schema):
    for split in ["train", "dev", "test"]:
        bad, total = collections.Counter(), 0
        for doc in load_split(split):
            for evs in events_by_sentence(doc).values():
                for ev in evs:
                    total += 1
                    for arg in ev["arguments"]:
                        if arg["argument_type"] not in schema[ev["event_type"]]["roles"]:
                            bad[(ev["event_type"], arg["argument_type"])] += 1
        print(f"{split}: {total} events, illegal roles: {dict(bad) or 0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--doc", type=int)
    ap.add_argument("--sent", type=int)
    ap.add_argument("--schemas", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    schema = load_schema()
    if a.schemas:
        print(render_schemas(schema))
    if a.check:
        check(schema)
    if a.doc is not None:
        doc = load_split(a.split)[a.doc]
        evs = events_by_sentence(doc)[a.sent]
        print(f"[Target Sentence]\n{doc['sentences'][a.sent]}\n")
        print(f"[Gold Events]\n{render_gold_events(evs, schema)}\n")
        print(f"[Inventory]\n{inventory(evs)}")


if __name__ == "__main__":
    main()
