import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _source_row(events):
    return {
        "id": "doc::sentence-0",
        "document_id": "doc",
        "split": "train",
        "mode": "sentence",
        "sentence_index": 0,
        "target_sentence": "we propose a method .",
        "sentences": ["we propose a method ."],
        "abstract": "we propose a method .",
        "target_events": events,
    }


def _event(event_type, offset=1):
    return {
        "event_type": event_type,
        "trigger": {"text": "propose", "offsets": [offset]},
        "arguments": [],
    }


def test_build_row_uses_cot_target_without_gold_leakage_in_prompt():
    from build_dataset import build_train_row

    cot_row = {
        "sample_id": "train:doc:0",
        "completion": "Reasoning:\nEvent 1: evidence. Type: <PRP></type>",
        "gold_event_type_inventory": "PRP: 1",
        "gold_arguments_by_event": "Event 1:\nEvent Type: PRP\nArguments:\n- Content: method",
        "qc": {"ok": True},
        "api_used": True,
    }
    row = build_train_row(cot_row, _source_row([_event("PRP")]))

    assert row["id"] == "doc::sentence-0"
    assert "gold_event_type_inventory" not in row["prompt"]
    assert "gold_arguments_by_event" not in row["prompt"]
    assert "Event Type Inventory:" in row["completion"]
    assert "PRP: 1" in row["completion"]
    assert row["target_type_inventory"] == {"PRP": 1}


def test_parser_prefers_final_inventory_and_falls_back_to_type_tags():
    from parser import parse_type_cot_output

    parsed = parse_type_cot_output(
        "Reasoning mentions Type: <FAC></type>\n\nFinal Answer:\nEvent Type Inventory:\nPRP: 2\nCMP: 1"
    )
    assert parsed["types"] == ["PRP", "PRP", "CMP"]
    assert parsed["source"] == "inventory"

    fallback = parse_type_cot_output("Event 1: x Type: <FIN></type>\nEvent 2: y Type: <FIN></type>")
    assert fallback["types"] == ["FIN", "FIN"]
    assert fallback["source"] == "type_tags"

    none = parse_type_cot_output("Final Answer:\nEvent Type Inventory:\nNone")
    assert none["types"] == []
    assert none["source"] == "inventory"


def test_evaluate_counts_duplicate_type_multiset():
    from evaluate import evaluate_rows

    data_rows = [
        _source_row([_event("FIN", 1), _event("FIN", 2), _event("CMP", 3)]),
        _source_row([]) | {"id": "doc::sentence-1"},
    ]
    predictions = [
        {"id": "doc::sentence-0", "raw_output": "Event Type Inventory:\nFIN: 1\nCMP: 1\nFAC: 1"},
        {"id": "doc::sentence-1", "raw_output": "Event Type Inventory:\nNone"},
    ]
    result = evaluate_rows(data_rows, predictions)

    assert result["type_multiset"]["tp"] == 2
    assert result["type_multiset"]["fn"] == 1
    assert result["type_multiset"]["fp"] == 1
