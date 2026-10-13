"""Ensure R0 changes training without changing the historical eval contract."""
import ast
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))


def test_grpo_imports_r0_and_keeps_evaluator_separate():
    tree=ast.parse((HERE/'train_grpo.py').read_text())
    imports=[node for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert any(node.module=='action_rewards' and any(x.name=='reward_action_r0' and x.asname=='reward_action' for x in node.names) for node in imports)
    assert not any(node.module=='action_evaluation' and any(x.name=='reward_action' for x in node.names) for node in imports)
    text=(HERE/'train_grpo.py').read_text()
    assert "reward_version=REWARD_VERSION" in text
    assert 'probability_statistics(current,old,mask)' in text
    assert 'normalized_structure' in text


def test_r0_reward_source_is_in_default_run_inventory():
    from action_common import source_files
    assert HERE/'action_rewards.py' in source_files()
    assert HERE/'grpo_diagnostics.py' in source_files()
