"""Parse Type-COT model outputs into event-type multisets."""

from __future__ import annotations

from collections import Counter
import re


EVENT_CODES = ("PUR", "ITT", "RWS", "RWF", "PRP", "WKS", "MDS", "FIN", "CMP", "FAC")
TYPE_RE = "|".join(EVENT_CODES)


def _counter_to_types(counts: Counter[str]) -> list[str]:
    output: list[str] = []
    for code in EVENT_CODES:
        output.extend([code] * counts[code])
    return output


def _parse_inventory_tail(tail: str) -> tuple[Counter[str] | None, list[str]]:
    unknown: list[str] = []
    text = tail.strip()
    if not text:
        return None, unknown
    if re.search(r"\bNone\b\.?", text) and not re.search(rf"\b(?:{TYPE_RE})\s*:", text):
        return Counter(), unknown
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.fullmatch(rf"<?({TYPE_RE})(?:></type>)?\s*:\s*<?(\d+)(?:></count>)?", stripped)
        if not match:
            prefix = stripped.split(":", 1)[0].strip("<> ")
            if prefix and prefix != "Final Answer":
                unknown.append(prefix)
            continue
        code, count = match.groups()
        counts[code] += int(count)
    return counts, unknown


def parse_type_cot_output(text: str) -> dict:
    """Return a tolerant parse result with duplicate event types preserved."""
    raw = text if isinstance(text, str) else ""
    result = {
        "types": [],
        "counts": {},
        "source": "empty" if not raw.strip() else "none",
        "unknown_lines": [],
        "type_tag_count": 0,
        "inventory_found": False,
    }
    inventory_match = re.search(r"Event Type Inventory:\s*(.*)", raw, re.S)
    if inventory_match:
        counts, unknown = _parse_inventory_tail(inventory_match.group(1))
        result["inventory_found"] = True
        result["unknown_lines"] = unknown
        if counts is not None:
            result["source"] = "inventory"
            result["counts"] = {code: counts[code] for code in EVENT_CODES if counts[code]}
            result["types"] = _counter_to_types(counts)
            return result

    tags = re.findall(rf"Type:\s*<({TYPE_RE})></type>?", raw)
    result["type_tag_count"] = len(tags)
    if tags:
        counts = Counter(tags)
        result["source"] = "type_tags"
        result["counts"] = {code: counts[code] for code in EVENT_CODES if counts[code]}
        result["types"] = _counter_to_types(counts)
        return result
    if re.fullmatch(r"\s*None\.?\s*", raw):
        result["source"] = "plain_none"
    return result
