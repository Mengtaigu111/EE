import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / 'scripts' / 'rewrite_prompt_dataset.py'


def run_builder(tmp_path, completion):
    source = tmp_path / 'source'
    source.mkdir()
    row = dict(id='doc::sentence-0', sentences=['we propose a method .'],
               sentence_index=0, target_sentence='we propose a method .',
               prompt='old', completion=completion,
               target_events=[dict(event_type='PRP'), dict(event_type='PRP')],
               target_type_inventory={'PRP': 2}, private_gold_marker='GOLD_ONLY')
    for split in ('train', 'dev', 'test'):
        (source / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
    template = tmp_path / 'prompt.txt'
    template.write_text('TASK\nFull Abstract:\n[S1] <sentence 1>\n\n'
                        'TARGET SENTENCE [S{k}]:\n<target sentence>\n\nType-COT output:\n')
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, str(SCRIPT), '--source', str(source),
                             '--template', str(template), '--output', str(output)],
                            capture_output=True, text=True)
    return result, output, row


def test_rewrite_preserves_targets_and_repeated_events_without_gold_leakage(tmp_path):
    completion = ('Reasoning:\n1. Event Family Reasoning:\nContribution.\n'
                  '2. Event Reasoning:\nEvent 1: we (Proposer). Type: <PRP></type>\n'
                  'Event 2: method (Content). Type: <PRP></type>\n'
                  'Final Answer:\nEvent Type Inventory:\nPRP: 2')
    result, output, original = run_builder(tmp_path, completion)
    assert result.returncode == 0, result.stderr
    for split in ('train', 'dev', 'test'):
        rewritten = json.loads((output / f'{split}.jsonl').read_text())
        assert {k: v for k, v in rewritten.items() if k != 'prompt'} == {
            k: v for k, v in original.items() if k != 'prompt'}
        assert '[S1] we propose a method .' in rewritten['prompt']
        assert 'TARGET SENTENCE [S1]:\nwe propose a method .' in rewritten['prompt']
        assert 'GOLD_ONLY' not in rewritten['prompt']
        assert completion not in rewritten['prompt']


def test_inconsistent_reasoning_counts_block_rewrite_before_output(tmp_path):
    completion = ('Reasoning:\n1. Event Family Reasoning:\nContribution.\n'
                  '2. Event Reasoning:\nEvent 1: evidence. Type: <PRP></type>\n'
                  'Final Answer:\nEvent Type Inventory:\nPRP: 2')
    result, output, _ = run_builder(tmp_path, completion)
    assert result.returncode != 0
    assert 'reasoning counts' in result.stderr
    assert not output.exists()
