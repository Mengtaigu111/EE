"""Generate dev trees with the resident model and frozen historical evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

from verify_sorted_first_run import fixed_evaluation_modules


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _coverage(rows, predictions):
    expected = [row['id'] for row in rows]
    actual = [row['id'] for row in predictions]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Empty data or duplicate data IDs')
    if len(set(actual)) != len(actual) or actual != expected:
        raise ValueError('Prediction ID coverage/order mismatch')


def _paths(prefix):
    return {name: str(Path(str(prefix) + suffix).resolve()) for name, suffix in {
        'predictions': '.predictions.jsonl', 'merged': '.merged.jsonl',
        'metrics': '.metrics.json', 'scievents': '.metrics.scievents.json',
        'docs': '.metrics.docs.jsonl', 'complete': '.complete.json'}.items()}


def _reserve(prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if any(Path(path).exists() for path in _paths(prefix).values()):
        raise FileExistsError(prefix)
    # A failed evaluation also remains reserved: no partial output can be overwritten.
    with Path(str(prefix) + '.reservation').open('x', encoding='utf-8') as stream:
        stream.write('exclusive evaluation output prefix\n')


def _score(rows, predictions, document_gold_path, paths):
    _coverage(rows, predictions)
    evaluator, merger = fixed_evaluation_modules()
    reparsed = merger.reparse_raw_predictions(predictions)
    merged = merger.merge_sentence_rows(rows, reparsed)
    gold = _read(document_gold_path)
    expected = [row['id'] for row in gold]
    ids = [row['id'] for row in merged]
    if len(set(expected)) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('Document gold/prediction ID coverage mismatch')
    metrics, scievents, documents = evaluator.evaluate(gold, merged)
    metrics.update(data=str(Path(document_gold_path).resolve()), predictions=paths['merged'], oracle=False,
                   paths=paths, diagnostics={
                       'sentence_count': len(predictions),
                       'invalid_sentences': sum(bool(p['invalid']) for p in reparsed),
                       'hit_max_new_tokens': sum(bool(p.get('hit_max_new_tokens')) for p in predictions)})
    _jsonl(paths['predictions'], predictions)
    _jsonl(paths['merged'], merged)
    _json(paths['scievents'], scievents)
    _jsonl(paths['docs'], documents)
    return metrics


def _finish(metrics, started):
    metrics['wall_seconds'] = time.monotonic() - started
    _json(metrics['paths']['metrics'], metrics)
    prediction_path = Path(metrics['paths']['predictions'])
    _json(metrics['paths']['complete'], {
        'status': 'completed', 'count': metrics['diagnostics']['sentence_count'],
        'wall_seconds': metrics['wall_seconds'], **metrics['diagnostics'],
        'output_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
    return metrics


def score_predictions(rows, predictions, document_gold_path, output_prefix):
    """Recompute full official offset scores from complete ordered raw predictions."""
    rows, predictions = list(rows), list(predictions)
    _coverage(rows, predictions)
    _reserve(output_prefix)
    started = time.monotonic()
    return _finish(_score(rows, predictions, document_gold_path, _paths(output_prefix)), started)


def evaluate_current_model(model, tokenizer, rows, document_gold_path, output_prefix,
                           batch_size=4, max_new_tokens=319):
    """Evaluate without loading a model or disturbing training mode or RNG state."""
    import numpy as np
    import torch
    source = Path(__file__).resolve().parent.parent / 'scripts/model_utils.py'
    spec = importlib.util.spec_from_file_location('online_dev_model_utils', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_tokens = module.prompt_tokens
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError('batch_size and max_new_tokens must be positive')
    rows = list(rows)
    _coverage(rows, rows)
    _reserve(output_prefix)
    started = time.monotonic()
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    training, cache = model.training, model.config.use_cache
    padding = {key: getattr(tokenizer, key) for key in ('padding_side', 'truncation_side') if hasattr(tokenizer, key)}
    predictions = []
    try:
        evaluator, merger = fixed_evaluation_modules()
        # The fixed merger imports the fixed parser, whose origin was checked above.
        parser = merger.parse_forest
        model.eval()
        model.config.use_cache = True
        tokenizer.padding_side = 'left'
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                encoded = [{'input_ids': prompt_tokens(row['prompt'], tokenizer)} for row in batch]
                if max(len(e['input_ids']) for e in encoded) + max_new_tokens > model.config.max_position_embeddings:
                    raise ValueError('Inference prompt exceeds model context; truncation forbidden')
                inputs = tokenizer.pad(encoded, padding=True, return_tensors='pt').to(model.device)
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    temperature=None, top_p=None, top_k=None)
                new_tokens = generated[:, inputs['input_ids'].shape[1]:]
                outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                if len(outputs) != len(batch) or len(new_tokens) != len(batch):
                    raise ValueError('Generation batch ID coverage mismatch')
                for row, raw, ids in zip(batch, outputs, new_tokens):
                    hit_limit = bool((ids != tokenizer.pad_token_id).sum().item() >= max_new_tokens
                                     and tokenizer.eos_token_id not in ids.tolist())
                    events, diagnostics = parser(raw)
                    record = {key: row[key] for key in ('id', 'document_id', 'split', 'mode',
                              'sentence_index', 'target_bounds', 'target_sentence') if key in row}
                    record.update(raw_output=raw, parsed_events=events, parse_diagnostics=diagnostics,
                                  invalid=diagnostics['invalid'], hit_max_new_tokens=hit_limit)
                    predictions.append(record)
        metrics = _score(rows, predictions, document_gold_path, _paths(output_prefix))
        return _finish(metrics, started)
    finally:
        model.train(training)
        model.config.use_cache = cache
        for key, value in padding.items():
            setattr(tokenizer, key, value)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
