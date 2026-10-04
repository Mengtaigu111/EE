#!/usr/bin/env python3
"""Gold-only diagnosis: fixed parser -> sentence merge, no scorer-based correspondence."""
import collections
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STAGE = HERE.parent
sys.path.insert(0, str(STAGE / 'curriculum'))
from verify_sorted_first_run import fixed_evaluation_modules

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def raw_event(e):
    return {'event_type': e['type'], 'trigger': {'tokens': e['trigger'].split()},
            'arguments': [{'argument_type': a['role'], 'tokens': a['text'].split()} for a in e['args']]}

def key(e):
    return json.dumps(e, sort_keys=True, ensure_ascii=False)

def scores(gold, pred, mode, names):
    import utils
    with contextlib.redirect_stdout(io.StringIO()):
        return dict(zip(names, map(float, utils.calculate_F1(gold, pred, mode))))

def audit():
    evaluation, merge = fixed_evaluation_modules()
    linear = sys.modules['linearize']; align = sys.modules['align_events']
    gold_path = PROJECT / 'data/SciEvents/test.json'
    rows_path = STAGE / 'data/sentence-c-appendix-no-copy-20261002/test.jsonl'
    prior_path = PROJECT / 'task/curriculum-handoff-20261004/04-score-consistency-metrics.json'
    paths = [gold_path, rows_path, prior_path, PROJECT / 'task/curriculum-handoff-20261004/04-score-consistency-audit.md', PROJECT / 'utils.py', Path(__file__),
             STAGE / 'curriculum/verify_sorted_first_run.py'] + [Path(sys.modules[n].__file__) for n in
             ('ontology', 'linearize', 'align_events', 'evaluate', 'merge_sentence_predictions')]
    if (HERE / 'instance_scorer.py').exists(): paths.append(HERE / 'instance_scorer.py')
    paths.append(HERE / 'ambiguity_resolver.py')
    before = {str(p): sha(p) for p in paths}
    docs = json.loads(gold_path.read_text()); by_doc = {d['doc_id']: d for d in docs}
    rows = [json.loads(s) for s in rows_path.read_text().splitlines() if s.strip()]
    assert len(docs) == len(by_doc) == 250 and len(rows) == len({r['id'] for r in rows}) == 1627
    grouped = collections.defaultdict(list)
    for row in rows: grouped[row['document_id']].append(row)
    assert set(grouped) == set(by_doc)
    parsed_rows = []; origins = {}; counts = collections.Counter(); losses = []; spans = []
    for doc_id, group in grouped.items():
        document = by_doc[doc_id]; tokens = document['document']
        original_index = collections.defaultdict(collections.deque)
        for i, e in enumerate(document['events']): original_index[key(e)].append(i)
        for mode, field in [('offset', 'offsets'), ('token', 'tokens')]:
            trigger_groups = collections.defaultdict(list)
            for ev in document['events']: trigger_groups[tuple(ev['trigger'][field])].append(ev)
            collisions = [es for es in trigger_groups.values() if len(es) > 1]
            counts[mode + '_duplicate_trigger_expression_groups'] += len(collisions)
            counts[mode + '_duplicate_trigger_expression_instances'] += sum(len(es) for es in collisions)
            counts[mode + '_docs_with_duplicate_trigger_expression'] += bool(collisions)
            if field == 'tokens':
                counts['same_trigger_tokens_distinct_offset_groups'] += sum(len({tuple(e['trigger']['offsets']) for e in es})>1 for es in collisions)
            for ev in document['events']:
                ac = collections.Counter(tuple(a[field]) for a in ev['arguments'])
                counts[mode + '_duplicate_argument_expression_groups'] += sum(n>1 for n in ac.values())
        targets = [e for r in group for e in r['target_events']]
        assert collections.Counter(map(key, targets)) == collections.Counter(map(key, document['events']))
        ordered_origins = []; all_parsed = []; scopes = []
        for row in sorted(group, key=lambda r: r['sentence_index']):
            assert row['tokens'] == tokens
            indexed = [(original_index[key(e)].popleft(), e) for e in row['target_events']]
            indexed.sort(key=lambda v: (min(v[1]['trigger']['offsets']), max(v[1]['trigger']['offsets']), v[1]['event_type']))
            canonical = linear.canonical_events({'document': tokens, 'events': row['target_events']})
            assert len(canonical) == len(indexed)
            for (_, source), can in zip(indexed, canonical):
                assert linear.canonical_events({'document': tokens, 'events': [source]})[0] == can
            parsed, diag = linear.parse_forest(row['completion'])
            text_only = [{'type': e['type'], 'trigger': e['trigger'], 'args': [{'role': a['role'], 'text': a['text']} for a in e['args']]} for e in canonical]
            assert parsed == text_only and not diag['invalid'], row['id']
            counts['canonical_completion_exact_sentence_rows'] += 1
            parsed_rows.append({'id': row['id'], 'raw_output': row['completion'], 'parsed_events': parsed,
                                'invalid': False, 'parse_diagnostics': diag})
            all_parsed.extend(parsed); scopes.extend([tuple(row['target_bounds'])] * len(parsed))
            ordered_origins.extend([(i, source, can, row['id'], tuple(row['target_bounds'])) for (i, source), can in zip(indexed, canonical)])
        aligned, stat = align.align_forest(tokens, document['sentences'], all_parsed,
                                          trigger_scopes=scopes, argument_sentence_preference=False)
        assert len(aligned) == len(ordered_origins), 'Dropped trigger requires explicit provenance handling'
        origins[doc_id] = (ordered_origins, aligned, all_parsed)
        for (event_index, source, can, sentence_id, scope), output in zip(ordered_origins, aligned):
            counts['events'] += 1
            assert source['event_type'] == output['event_type']
            clean_source = evaluation.strip_private([source])[0]
            clean_output = evaluation.strip_private([output])[0]
            counts['event_fields_exact_ignoring_argument_order_and_nugget_metadata'] += (
                clean_source['trigger'] == clean_output['trigger'] and
                collections.Counter(map(key, clean_source['arguments'])) == collections.Counter(map(key, clean_output['arguments'])))
            # Canonical argument sorting is independently reconstructed on ORIGINAL identities.
            indexed_args = list(enumerate(source['arguments']))
            indexed_args.sort(key=lambda v: (linear.role_index(source['event_type'], v[1]['argument_type']),
                                           min(v[1]['offsets']), max(v[1]['offsets'])))
            assert len(output['arguments']) == len(indexed_args), 'Dropped argument requires explicit provenance handling'
            items = [('trigger', None, source['trigger'], can['trigger'], can['trigger_offsets'], output['trigger'], scope)]
            for (ai, original), ca, out in zip(indexed_args, can['args'], output['arguments']):
                assert original['argument_type'] == ca['role'] == out['argument_type']
                items.append(('argument', ai, original, ca['text'], ca['offsets'], out, None))
            for kind, ai, original, canonical_text, canonical_offsets, out, search_scope in items:
                counts[kind + '_total'] += 1
                counts[kind + '_original_tokens_match_document_ordered_offsets'] += original['tokens'] == [tokens[i] for i in original['offsets']]
                counts[kind + '_original_text_matches_token_field'] += original.get('text') == ' '.join(original['tokens'])
                exact = original['offsets'] == out['offsets']
                original_text = ' '.join(original['tokens'])
                candidates, method = align._candidates(tokens, [t.casefold() for t in tokens], canonical_text, scope=search_scope)
                categories = []
                if original['offsets'] != out['offsets'] and canonical_offsets == out['offsets']:
                    categories.append('position_mismatch_only_original_order_or_multiplicity_loss')
                if original['offsets'] != canonical_offsets:
                    categories.append('canonical_sorted_unique_loses_original_offset_sequence')
                    if len(original['offsets']) != len(set(original['offsets'])): categories.append('canonical_unique_loses_multiplicity')
                    if original['offsets'] != sorted(original['offsets']): categories.append('reverse_or_nonmonotonic_original_order')
                if canonical_text != original_text: categories.append('canonical_text_differs_from_original_tokens')
                if original.get('text') != original_text: categories.append('original_text_field_differs_from_tokens')
                if not exact:
                    categories.append('wrong_' + kind + '_position')
                    if out['tokens'] == original['tokens']: categories.append('same_tokens_different_offsets')
                    if out['text'] == original.get('text'): categories.append('same_text_field_different_offsets')
                    if canonical_offsets != out['offsets']:
                        categories.append('canonical_sorted_span_position_mismatch')
                        if canonical_offsets in candidates: categories.append('gold_sorted_span_in_candidates_but_not_selected')
                        else:
                            categories.append('gold_sorted_span_absent_from_candidates')
                            if method.startswith('subsequence'): categories.append('valid_gold_subsequence_omitted_by_greedy_enumeration')
                    if len(candidates) > 1: categories.append('multiple_same_text_candidates')
                    if len(canonical_offsets)>1 and any(b != a+1 for a,b in zip(canonical_offsets,canonical_offsets[1:])):
                        categories.append('discontinuous_gold')
                        if method.startswith('exact'): categories.append('contiguous_candidate_suppresses_discontinuous_search')
                for cat in categories: counts[kind + '_' + cat] += 1
                counts[kind + '_offset_sequence_exact'] += exact
                counts[kind + '_tokens_exact'] += out['tokens'] == original['tokens']
                counts[kind + '_text_field_exact'] += out['text'] == original.get('text')
                counts[kind + '_canonical_sorted_offsets_exact'] += out['offsets'] == canonical_offsets
                counts[kind + '_canonical_text_exact'] += out['text'] == canonical_text
                if categories:
                    record = {'doc_id': doc_id, 'title': document['title'], 'sentence_id': sentence_id,
                              'event_index': event_index, 'event_type': source['event_type'], 'kind': kind,
                              'argument_index': ai, 'original': original, 'canonical_text': canonical_text,
                              'canonical_offsets': canonical_offsets, 'aligned': out, 'candidate_method': method,
                              'candidates': candidates, 'categories': categories, 'trigger_original': source['trigger'],
                              'trigger_aligned': output['trigger'], 'context_tokens': tokens}
                    spans.append(record)
                    if not exact: losses.append(record)
    counts['documents_with_position_mismatch'] = len({r['doc_id'] for r in losses})
    counts['documents_with_trigger_position_mismatch'] = len({r['doc_id'] for r in losses if r['kind']=='trigger'})
    counts['documents_with_argument_position_mismatch'] = len({r['doc_id'] for r in losses if r['kind']=='argument'})
    merged = merge.merge_sentence_rows(rows, parsed_rows, deduplicate=False)
    merged_default = merge.merge_sentence_rows(rows, parsed_rows, deduplicate=True)
    merged_by_id = {d['id']: evaluation.strip_private(d['aligned_events']) for d in merged}
    for doc_id, (_, aligned, _) in origins.items():
        assert collections.Counter(map(key, evaluation.strip_private(aligned))) == collections.Counter(map(key, merged_by_id[doc_id]))
    counts['default_dedup_events_removed'] = sum(len(d['aligned_events']) for d in merged) - sum(len(d['aligned_events']) for d in merged_default)
    gold = [d['events'] for d in docs]
    raw = [[raw_event(e) for e in origins[d['doc_id']][2]] for d in docs]
    roundtrip = [merged_by_id[d['doc_id']] for d in docs]
    metrics = {'original_gold_self_offset': scores(gold, gold, 'offset', evaluation.METRIC_NAMES),
               'original_gold_self_token': scores(gold, gold, 'token', evaluation.METRIC_NAMES),
               'gold_completion_raw_token': scores(gold, raw, 'token', evaluation.METRIC_NAMES),
               'gold_completion_roundtrip_offset': scores(gold, roundtrip, 'offset', evaluation.METRIC_NAMES),
               'gold_completion_roundtrip_token': scores(gold, roundtrip, 'token', evaluation.METRIC_NAMES)}
    prior = json.loads(prior_path.read_text())['gold_self']
    for mode in ('offset','token'):
        for metric in ('TI','TC','AI','AC','EC_I','EC_C'):
            for suffix, old_name in [('f1','F1'),('p','precision'),('r','recall')]:
                assert metrics['original_gold_self_'+mode][metric+'_'+suffix] == prior[mode][metric][old_name]
    from ambiguity_resolver import resolve_span
    resolver_records = []
    for record in losses:
        if 'gold_sorted_span_absent_from_candidates' not in record['categories']: continue
        resolved = resolve_span(record['context_tokens'], record['canonical_text'])
        covered = record['canonical_offsets'] in resolved['candidates']
        counts['new_resolver_legacy_omission_cases'] += 1
        counts['new_resolver_legacy_omission_gold_candidate_covered'] += covered
        counts['new_resolver_legacy_omission_status_' + resolved['status']] += 1
        counts['new_resolver_max_candidates_observed'] = max(counts['new_resolver_max_candidates_observed'], len(resolved['candidates']))
        resolver_records.append({'doc_id':record['doc_id'], 'event_index':record['event_index'],
                                 'argument_index':record['argument_index'], 'canonical_text':record['canonical_text'],
                                 'gold_offsets_diagnostic_only':record['canonical_offsets'],
                                 'legacy_candidate_method':record['candidate_method'],
                                 'gold_candidate_covered':covered, 'resolved':resolved})
    assert len(resolver_records)==40 and all(r['gold_candidate_covered'] for r in resolver_records)
    assert all(r['resolved']['status']=='ambiguous' and r['resolved']['offsets'] is None for r in resolver_records)
    fixtures = {}
    reverse = {'document': ['a','b'], 'events': [{'event_type': 'PUR', 'trigger': {'text':'b a','tokens':['b','a'],'offsets':[1,0]}, 'arguments': []}]}
    fixtures['canonical_reverse_loss'] = {'input': reverse, 'canonical': linear.canonical_events(reverse)}
    fixtures['subsequence_incomplete_enumeration'] = {'tokens': ['a','b','b','c'], 'query':['a','b','c'],
        'returned': align._subsequences(['a','b','b','c'], ['a','b','c'], 0, 4), 'also_valid_but_omitted':[0,2,3],
        'interpretation': 'Candidate enumeration is greedy per start and incomplete; text alone cannot prefer omitted gold span.'}
    from_instance = None
    scorer_path = HERE / 'instance_scorer.py'
    if scorer_path.exists():
        import importlib.util
        spec=importlib.util.spec_from_file_location('gold_audit_instance', scorer_path); mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        try:
            from_instance = {label: mod.score_documents(gold, pred, mode=mode) for label,pred,mode in
                [('original_gold_self_offset',gold,'offset'),('original_gold_self_token',gold,'token'),
                 ('gold_completion_raw_token',raw,'token'),('gold_completion_roundtrip_offset',roundtrip,'offset'),
                 ('gold_completion_roundtrip_token',roundtrip,'token')]}
        except (TypeError, AttributeError) as exc: from_instance={'api_not_ready':str(exc)}
    if from_instance and 'api_not_ready' not in from_instance:
        assert all(value == 1.0 for label in ('original_gold_self_offset','original_gold_self_token')
                   for value in from_instance[label]['metrics'].values())
        assert from_instance['gold_completion_raw_token']['metrics'] == from_instance['gold_completion_roundtrip_token']['metrics']
    assert before == {str(p): sha(p) for p in paths}
    result = {'scope':'full test gold diagnostic ONLY; not model predictions; no gold-guided repair',
        'coverage': {'documents':len(docs), 'sentence_rows':len(rows), 'document_ids_exact':True,
                     'original_target_events_multiset_exact':True, 'canonical_completion_order_exact':True,
                     'previous_gold_self_all_18_metrics_exactly_reproduced':True,
                     'aligned_merge_multiset_exact_deduplicate_false':True},
        'source_hashes':before, 'fixed_origins':{n:sys.modules[n].__file__ for n in ('ontology','linearize','align_events','evaluate','merge_sentence_predictions')},
        'metrics_official':metrics,'counts':dict(counts),'position_mismatch_records':losses,
        'all_representation_difference_records':spans,'fixtures':fixtures,'metrics_instance':from_instance,
        'gold_free_resolver_legacy_omission_diagnostic':resolver_records}
    (HERE/'gold-roundtrip-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = ['# Gold sentence-tree roundtrip audit', '',
        'Full test: 250 documents / 1,627 sentence completions. Fixed evaluation-v2 parser and aligner; merge deduplicate=False. Gold is diagnostic only.', '',
        'Correspondence never uses a global scorer: exact original target-event multiset coverage, stable canonical event ordering, canonical original argument ordering, exact parsed completion equality, and pre-sort aligner ordering are asserted. Merge output multiset is checked against that independently paired aligner output.', '',
        '## All official metrics (%)', '', '| input | metric | F1 | precision | recall |','|---|---|---:|---:|---:|']
    for label,m in metrics.items():
        for metric in ('TI','TC','AI','AC','EC_I','EC_C'):
            lines.append(f'| {label} | {metric} | {m[metric+"_f1"]*100:.6f} | {m[metric+"_p"]*100:.6f} | {m[metric+"_r"]*100:.6f} |')
    if from_instance and 'api_not_ready' not in from_instance:
        lines += ['', '## Instance-preserving metrics (%)', '', '| input | metric | F1 | precision | recall |','|---|---|---:|---:|---:|']
        for label, scored in from_instance.items():
            m = scored['metrics']
            for metric in ('TI','TC','AI','AC','EC_I','EC_C'):
                lines.append(f'| {label} | {metric} | {m[metric+"_f1"]*100:.6f} | {m[metric+"_p"]*100:.6f} | {m[metric+"_r"]*100:.6f} |')
    lines += ['', '## Proven losses and boundaries', '',
        f'Ordered-offset mismatches: {counts["trigger_total"]-counts["trigger_offset_sequence_exact"]} triggers and {counts["argument_total"]-counts["argument_offset_sequence_exact"]} arguments. Of the argument mismatches, {counts["argument_position_mismatch_only_original_order_or_multiplicity_loss"]} preserve the complete sorted gold span and differ only because canonicalization removed order or multiplicity. Actual sorted-span mismatches: {counts["argument_canonical_sorted_span_position_mismatch"]} arguments. Canonical sequence loss affects {counts["argument_canonical_sorted_unique_loses_original_offset_sequence"]} arguments in total; categories overlap.', '',
        'Original gold self-scoring reproduces the existing dictionary-overwrite defect. Raw gold completions also lose original reverse/nonmonotonic token order: canonical_events sorts and uniquifies offsets before producing text. This is an information-loss bug for an offset-sequence-preserving evaluation contract, not an alignment error that text can undo.', '',
        'The aligner additionally chooses among repeated same-text spans, prefers contiguous occurrences before any discontinuous candidates, greedily enumerates subsequences, and prefers argument text matching a trigger. These policies can select a different position even when text is correct. Repeated text, same-position event multiplicity, discontinuity and original reverse order are frequently unidentifiable from this completion schema. A wrong gold position alone is not proof that a gold-free deterministic alternative could recover it.', '',
        'The subsequence fixture proves incomplete candidate enumeration, but its omitted span is also text-ambiguous. Enumerating all subsequences removes that algorithmic limitation without proving a uniquely correct location.', '',
        'The new independent gold-free resolver enumerates contiguous and noncontiguous candidates together. With the default 256-candidate cap, it covers every gold span in all 40 legacy omitted-candidate cases (36 greedy-subsequence omissions and 4 contiguous-preference suppressions). All 40 resolve to ambiguous, none truncate; maximum observed candidate count is 141. It returns offsets=None in every case. This proves improved candidate completeness, not recovered positions or higher real-model scores. No actual prediction or historical aligner is changed.', '',
        'Do not add gold-self loss to real model scores. Official scorer collisions can undercount and overcount; token text can match the wrong location. These measurements establish loss mechanisms, not the net bias in actual predictions.', '',
        '## Counts (categories overlap)', '', '```json', json.dumps(dict(counts),ensure_ascii=False,indent=2), '```', '',
        '## Gold-free repair options', '',
        '- Preserve original offset order when constructing supervised text targets; existing completions cannot recover the removed order.',
        '- For future model outputs, serialize explicit ordered token indices or span IDs and event-instance IDs. Validate them against input tokens without consulting gold.',
        '- Retain all alignment candidates and mark ambiguity; evaluate text quality alongside offset quality. If choosing candidates, freeze a policy using dev only and report tradeoffs.',
        '- Use instance-preserving one-to-one scoring for collisions and retain old official scores for traceability. Avoid deduplication when scoring event-instance multiplicity.', '',
        '## Actual examples', '']
    seen=set()
    for rec in spans:
        cats=set(rec['categories'])-seen
        if not cats: continue
        seen.update(cats)
        example={k:v for k,v in rec.items() if k!='context_tokens'}
        lines += ['```json',json.dumps(example,ensure_ascii=False,indent=2),'```','']
    lines += ['## Reproduction', '',
        '`/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/stage2-event-tree/scorer-repair-20261004/audit_gold_roundtrip.py`', '',
        'CPU only. JSON contains all position mismatch records, all representation differences, full contexts, exact source SHA-256, fixed module origins and all 18 official metrics for each condition. The script asserts sources unchanged after scoring. No aligner, scorer, old data or real predictions are modified.']
    (HERE/'gold-roundtrip-report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'coverage':result['coverage'],'counts':dict(counts),'metrics_official':metrics},ensure_ascii=False,indent=2))

if __name__=='__main__': audit()
