"""Reuse the independently validated supplementary test artifact checks.

Adaptations only: this run's tree/checkpoints directory and measured batch size.
Original verifier source remains untouched; its hash is saved in setup.
"""
import contextlib,io,math,sys
from pathlib import Path
ROOT=Path('/home/myluo/EE/EXCEEDS')
sys.path.insert(0,str(ROOT/'tasks/stage2-event-tree/curriculum'))
from online_dev_config import jsonl,read,require,sha,verify_locks
from online_dev_trainer import validate_online_checkpoint
from verify_sorted_first_run import fixed_evaluation_modules
from run_sorted_first_online_dev import test_diagnostics
COMPLEX={'Consecutive_AC_f1','Overlap_TC_f1','Overlap_AC_f1','Reverse_AC_f1','Subevent_TC_f1','Subevent_AC_f1','Subevent_EC_f1'}

def _scores(actual, expected, label):
    require(set(actual) == set(expected), f'{label}: metric inventory differs')
    for name, value in actual.items():
        require(isinstance(value, (int, float)) and math.isfinite(value) and
                math.isclose(value, expected[name], rel_tol=1e-12, abs_tol=1e-12),
                f'{label}: {name} differs')

def _artifacts(recipe, original, step, predictions_path, merged_path, metrics_path,
               expected_official, expected_scores, expected_complex, expected_generation=None, batch_size=4):
    """Reconstruct every score from original raw text, then compare saved artifacts."""
    prediction = Path(predictions_path)
    merged_path, metrics_path = Path(merged_path), Path(metrics_path)
    rows = jsonl(Path(recipe['data_dir'])/'test.jsonl')
    gold = jsonl(Path(recipe['document_data_dir'])/'test.jsonl')
    raw = jsonl(prediction)
    ids = [r['id'] for r in rows]
    require(len(ids) == len(set(ids)) == len(raw) == 1627, 'Test sentence coverage/count differs')
    require([p['id'] for p in raw] == ids, 'Test predictions have missing/duplicate/unknown/reordered IDs')
    require(len(gold) == len({r['id'] for r in gold}) == 250, 'Test document gold coverage differs')
    for row, result in zip(rows, raw):
        for key in ('document_id', 'split', 'mode', 'sentence_index', 'target_bounds', 'target_sentence'):
            require(result.get(key) == row.get(key), 'Prediction metadata differs: '+key)
    generation = read(str(prediction)+'.complete.json')
    require(generation['status'] == 'completed' and generation['count'] == 1627 and
            generation['output_sha256'] == sha(prediction), 'Prediction completion/hash differs')
    require(generation['hit_max_new_tokens'] == sum(bool(p.get('hit_max_new_tokens')) for p in raw),
            'Generation truncation count differs')
    if expected_generation is not None:
        require(generation == expected_generation, 'State generation record differs')
    config = read(str(prediction)+'.config.json')
    checkpoint = original/'tree/checkpoints'/f'checkpoint-{step}'
    validate_online_checkpoint(checkpoint, step, 3208)
    require(Path(config['adapter']).resolve() == checkpoint.resolve(), 'Generation used wrong checkpoint')
    require(config['adapter_sha256'] == sha(checkpoint/'adapter_model.safetensors'), 'Generated adapter hash differs')
    require(config['data_path'] == str((Path(recipe['data_dir'])/'test.jsonl').resolve()) and
            config['data_sha256'] == sha(Path(recipe['data_dir'])/'test.jsonl'), 'Generation test input differs')
    for key, value in [('seed', 42), ('batch_size', batch_size), ('max_new_tokens', 319),
                       ('do_sample', False), ('enable_thinking', False), ('precision', 'bf16'),
                       ('limit', None), ('model_path', recipe['model_path'])]:
        require(config[key] == value, 'Generation config differs: '+key)
    verify_locks({str(Path(recipe['model_path'])/name): digest
                  for name, digest in config['model_files'].items()})
    evaluator, merger = fixed_evaluation_modules()
    reparsed = merger.reparse_raw_predictions(raw)
    for old, rebuilt in zip(raw, reparsed):
        for key in ('parsed_events', 'parse_diagnostics', 'invalid'):
            require(old[key] == rebuilt[key], 'Raw fixed-parser output differs: '+key)
    merged = merger.merge_sentence_rows(rows, reparsed)
    require(len(merged) == len({r['id'] for r in merged}) == 250 and
            {r['id'] for r in merged} == {r['id'] for r in gold}, 'Merged document coverage differs')
    require(merged == jsonl(merged_path), 'Saved merge differs from reconstructed raw predictions')
    with contextlib.redirect_stdout(io.StringIO()):
        metrics, documents, per_doc = evaluator.evaluate(gold, merged)
    saved = read(metrics_path)
    require(metrics['scope'] == 'full' and metrics['documents'] == 250, 'Official test scope is incomplete')
    require(set(metrics['official']) == set(evaluator.METRIC_NAMES) and len(metrics['official']) == 18,
            'Official metric inventory differs')
    _scores(metrics['official'], saved['official'], 'Saved official test')
    _scores(metrics['official'], expected_official, 'State official test')
    for key in ('documents', 'scope', 'alignment', 'per_event_type', 'gold_events', 'predicted_events'):
        require(metrics[key] == saved[key], 'Saved official diagnostics differ: '+key)
    scievents_path = metrics_path.with_suffix('.scievents.json')
    require(documents == read(scievents_path), 'Saved SciEvents predictions differ')
    require(per_doc == jsonl(metrics_path.with_suffix('.docs.jsonl')), 'Saved per-document details differ')
    # This runs only the official CPU complex scorer on the verified same predictions.
    complex_scores = evaluator.run_complex_eval(scievents_path, sys.executable)
    require(complex_scores['exit_code'] == 0 and set(complex_scores['scores']) == COMPLEX,
            'Complex metric inventory or subprocess status differs')
    for label, stored in [('Saved complex', saved['complex']), ('State complex', expected_complex)]:
        require(stored['exit_code'] == 0, label+': recorded failure')
        _scores(complex_scores['scores'], stored['scores'], label)
    with contextlib.redirect_stdout(io.StringIO()):
        diagnostics = test_diagnostics(recipe, prediction, merged_path)
    require(diagnostics == expected_scores, 'Four-mode diagnostic scores/counts differ')
    _scores(diagnostics['official_offset'], metrics['official'], 'Four-mode official offset')
    return {'status': 'passed', 'step': step, 'sentences': 1627, 'documents': 250,
            'official_metrics': 18, 'complex_metrics': 7, 'score_modes': 4,
            'raw_reparsed_and_merged': True, 'completion_hash_verified': True,
            'official': metrics['official'], 'predictions': str(prediction)}
