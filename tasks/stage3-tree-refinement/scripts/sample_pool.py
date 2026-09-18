"""D1 candidate-pool normalization and deterministic merge helpers."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

from common import align_forest, event_key, event_key_string, read_jsonl, trigger_key
from diagnostics import label_candidate


def candidate_from_aligned_event(event: dict[str, Any]) -> dict[str, Any]:
    """Convert Stage-2 aligned output into the Stage-3 candidate shape."""
    trigger = event['trigger']
    return {
        'type': event['event_type'],
        'trigger_offsets': list(trigger['offsets']),
        'trigger_text': trigger.get('text', ' '.join(trigger.get('tokens', []))),
        'args': [
            {'role': argument['argument_type'],
             'offsets': list(argument['offsets']),
             'text': argument.get('text', ' '.join(argument.get('tokens', [])))}
            for argument in event.get('arguments', [])
        ],
    }


def align_text_events(row: dict[str, Any], events: Iterable[dict[str, Any]],
                      trigger_scopes: list[tuple[int, int]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align parsed text events with the unchanged T0 document policy."""
    aligned, stats = align_forest(
        row['tokens'], row['sentences'], list(events),
        trigger_scopes=trigger_scopes,
        argument_sentence_preference=True,
    )
    return [candidate_from_aligned_event(event) for event in aligned], stats


def _event_order(event: dict[str, Any]) -> tuple[Any, ...]:
    trigger = trigger_key(event)
    return trigger[0], trigger[-1], event['type'], event_key(event)


def forest_signature(events: Iterable[dict[str, Any]]) -> tuple[Any, ...]:
    """Return a source-order-independent forest identity for diversity counts."""
    return tuple(sorted((event_key(event) for event in events), key=repr))


def merge_document_candidates(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate candidates and count support over distinct c0..c5 sources."""
    merged: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    for source_row in sources:
        source = str(source_row['source'])
        seen_in_source = set()
        for event in source_row.get('events', []):
            key = event_key(event)
            if key in seen_in_source:
                continue
            seen_in_source.add(key)
            row = merged.get(key)
            if row is None:
                row = deepcopy(event)
                row.update({
                    'event_key': event_key_string(event),
                    'trigger_key': list(trigger_key(event)),
                    'support': 0,
                    'sources': [],
                })
                merged[key] = row
            if source not in row['sources']:
                row['sources'].append(source)
                row['sources'].sort()
                row['support'] = len(row['sources'])
    return sorted(merged.values(), key=_event_order)


def build_pool_row(document_id: str, sources: Iterable[dict[str, Any]],
                   gold_events: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    sources = list(sources)
    candidates = merge_document_candidates(sources)
    gold_events = list(gold_events)
    for candidate in candidates:
        candidate['label'] = label_candidate(candidate, gold_events)
    return {
        'document_id': document_id,
        'candidates': candidates,
        'forest_count': len({forest_signature(source.get('events', [])) for source in sources}),
    }


def oracle_candidates(candidates: Iterable[dict[str, Any]],
                      gold_events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-event-score candidate for each gold type/trigger pair."""
    candidates = list(candidates)
    gold_events = list(gold_events)
    by_pair: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not candidate.get('label', label_candidate(candidate, gold_events))['exists']:
            continue
        pair = (candidate['type'], trigger_key(candidate))
        by_pair.setdefault(pair, []).append(candidate)
    selected = []
    wanted = {(gold['event_type'], trigger_key(gold)) for gold in gold_events}
    for pair in sorted(wanted):
        options = by_pair.get(pair, [])
        if options:
            selected.append(max(options, key=lambda row: (
                row.get('label', {}).get('event_score', 0.0),
                row.get('label', {}).get('arg_r', 0.0),
                row.get('label', {}).get('arg_p', 0.0),
                repr(event_key(row)),
            )))
    return sorted(selected, key=_event_order)


def candidate_to_aligned_event(candidate: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    def text(offsets: list[int], fallback: str = '') -> str:
        return fallback or ' '.join(tokens[index] for index in offsets)

    trigger_offsets = list(candidate['trigger_offsets'])
    return {
        'event_type': candidate['type'],
        'trigger': {'text': text(trigger_offsets, candidate.get('trigger_text', '')),
                    'tokens': [tokens[index] for index in trigger_offsets],
                    'offsets': trigger_offsets},
        'arguments': [
            {'argument_type': argument['role'],
             'text': text(argument['offsets'], argument.get('text', '')),
             'tokens': [tokens[index] for index in argument['offsets']],
             'offsets': list(argument['offsets'])}
            for argument in candidate.get('args', [])
        ],
    }


def _official_for_candidates(pool_rows: Iterable[dict[str, Any]], data_by_id: dict[str, dict[str, Any]],
                             candidates_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    from evaluate import official_scores

    golds, predictions = [], []
    for pool_row in pool_rows:
        document_id = pool_row['document_id']
        golds.append(data_by_id[document_id]['gold_events'])
        predictions.append([
            candidate_to_aligned_event(candidate, data_by_id[document_id]['tokens'])
            for candidate in candidates_by_row.get(document_id, [])
        ])
    return official_scores(golds, predictions)


def summarize_pool(pool_rows: Iterable[dict[str, Any]], data_rows: Iterable[dict[str, Any]],
                   support_min: int = 1) -> dict[str, Any]:
    """Compute D1 recall/precision/oracle summaries without model inference."""
    pool_rows = list(pool_rows)
    data_by_id = {row['id']: row for row in data_rows}
    if set(data_by_id) != {row['document_id'] for row in pool_rows}:
        raise ValueError('pool/data document IDs do not match')
    selected_by_doc = {
        row['document_id']: [candidate for candidate in row['candidates']
                             if candidate.get('support', 0) >= support_min]
        for row in pool_rows
    }
    gold_trigger_count = 0
    covered_trigger_count = 0
    candidate_count = 0
    covered_candidates = 0
    gold_pair_count = 0
    covered_pair_count = 0
    oracle_by_doc = {}
    diversity = []
    for pool_row in pool_rows:
        gold = data_by_id[pool_row['document_id']]['gold_events']
        gold_triggers = {trigger_key(event) for event in gold}
        gold_pairs = {(event['event_type'], trigger_key(event)) for event in gold}
        selected = selected_by_doc[pool_row['document_id']]
        predicted_triggers = {trigger_key(event) for event in selected}
        predicted_pairs = {(event['type'], trigger_key(event)) for event in selected}
        gold_trigger_count += len(gold_triggers)
        covered_trigger_count += len(gold_triggers & predicted_triggers)
        gold_pair_count += len(gold_pairs)
        covered_pair_count += len(gold_pairs & predicted_pairs)
        candidate_count += len(selected)
        covered_candidates += sum(bool(candidate.get('label', {}).get('exists')) for candidate in selected)
        oracle_by_doc[pool_row['document_id']] = oracle_candidates(selected, gold)
        diversity.append(pool_row.get('forest_count', 0))
    precision = covered_candidates / candidate_count if candidate_count else 0.0
    recall = covered_trigger_count / gold_trigger_count if gold_trigger_count else 0.0
    pair_recall = covered_pair_count / gold_pair_count if gold_pair_count else 0.0
    result = {
        'support_min': support_min,
        'documents': len(pool_rows),
        'events': candidate_count,
        'trigger_recall': recall,
        'trigger_precision': precision,
        'type_trigger_recall': pair_recall,
        'mean_unique_forests': sum(diversity) / len(diversity) if diversity else 0.0,
    }
    result['official'] = _official_for_candidates(pool_rows, data_by_id, selected_by_doc)
    result['oracle_official'] = _official_for_candidates(pool_rows, data_by_id, oracle_by_doc)
    return result


def _load_model(model_path: str, adapter: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from model_utils import load_model_config

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting bf16 is required')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=load_model_config(model_path), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    model.config.use_cache = True
    return torch, tokenizer, model, set_seed


def _sample_sources(row: dict[str, Any], greedy_prediction: dict[str, Any], model: Any,
                    tokenizer: Any, torch: Any, prompt_tokens: Any, parse_forest: Any,
                    sample_count: int, temperature: float, top_p: float,
                    max_new_tokens: int) -> list[dict[str, Any]]:
    sources = []
    greedy_events, greedy_stats = align_text_events(row, greedy_prediction['parsed_events'])
    sources.append({'source': 'c0', 'events': greedy_events, 'alignment': greedy_stats})
    encoded = tokenizer.pad([{'input_ids': prompt_tokens(row['prompt'], tokenizer)}],
                            padding=True, return_tensors='pt').to('cuda')
    prompt_width = encoded['input_ids'].shape[1]
    if prompt_width + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(f"{row['id']}: inference context exceeds model limit")
    generated = model.generate(
        **encoded, do_sample=True, temperature=temperature, top_p=top_p,
        num_return_sequences=sample_count, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    for sample_index, ids in enumerate(generated[:, prompt_width:]):
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        events, diag = parse_forest(raw)
        aligned, stats = align_text_events(row, events)
        sources.append({'source': f'c{sample_index + 1}', 'events': aligned,
                        'raw_output': raw, 'parse_diagnostics': diag,
                        'alignment': stats,
                        'hit_max_new_tokens': bool(tokenizer.eos_token_id not in ids.tolist())})
    return sources


def _sample_main(args: argparse.Namespace) -> None:
    from linearize import parse_forest
    from model_utils import prompt_tokens, read_rows

    rows = read_rows(args.data)
    greedy_rows = {row['id']: row for row in read_jsonl(args.greedy)}
    if set(greedy_rows) != {row['id'] for row in rows}:
        raise ValueError('greedy prediction IDs do not match data')
    torch, tokenizer, model, set_seed = _load_model(args.model, args.adapter)
    set_seed(args.seed)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pool_rows = []
    with output.open('x', encoding='utf-8') as stream, torch.inference_mode():
        for row in rows:
            sources = _sample_sources(row, greedy_rows[row['id']], model, tokenizer, torch,
                                      prompt_tokens, parse_forest, args.sample_count,
                                      args.temperature, args.top_p, args.max_new_tokens)
            pool_row = build_pool_row(row['id'], sources, row['gold_events'])
            pool_row.update({'split': row.get('split'), 'sources': sources,
                             'wall_seconds': time.time() - started})
            stream.write(json.dumps(pool_row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            pool_rows.append(pool_row)
    summaries = [summarize_pool(pool_rows, rows, support_min=value) for value in (1, 2, 3)]
    summary = {'status': 'completed', 'documents': len(pool_rows),
               'wall_seconds': time.time() - started, 'summaries': summaries}
    greedy_rows_for_summary = []
    for pool in pool_rows:
        greedy_rows_for_summary.append({**pool, 'candidates': [candidate for candidate in pool['candidates']
                                    if 'c0' in candidate.get('sources', [])]})
    greedy_summary = summarize_pool(greedy_rows_for_summary, rows, support_min=1)
    summary['greedy_summary'] = greedy_summary
    if (rows and rows[0].get('split') == 'dev'
            and summaries[0]['trigger_recall'] - greedy_summary['trigger_recall'] < 0.03):
        supplement_rows = []
        for row in rows:
            sources = _sample_sources(row, greedy_rows[row['id']], model, tokenizer, torch,
                                      prompt_tokens, parse_forest, 10, 1.0, 0.95, args.max_new_tokens)
            supplement_rows.append(build_pool_row(row['id'], sources, row['gold_events']))
        supplement_path = output.with_name(output.stem + '.supplement-10.jsonl')
        with supplement_path.open('x', encoding='utf-8') as stream:
            for row in supplement_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        summary['supplement_10_temp1'] = summarize_pool(supplement_rows, rows, support_min=1)
    summary_path = output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--greedy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-count', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-new-tokens', type=int, default=768)
    args = parser.parse_args()
    if 'TMUX' not in os.environ:
        raise RuntimeError('GPU execution must run in tmux')
    if args.sample_count < 1 or args.max_new_tokens < 1:
        parser.error('sample-count and max-new-tokens must be positive')
    _sample_main(args)


if __name__ == '__main__':
    main()
