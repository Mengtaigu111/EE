"""Exercise interruption of the trainer's actual optimizer commit statements."""
import ast
import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('train_grpo_recoverable.py')


def trainer_tree():
    return ast.parse(SOURCE.read_text(), filename=str(SOURCE))


class CommitBoundaryTests(unittest.TestCase):
    def test_every_interrupt_keeps_updated_state_dirty_until_payload_and_rng_match(self):
        tree = trainer_tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and any(
                    isinstance(item, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'optimizer_dirty'
                        for target in item.targets) for item in node.body):
                body = node.body
                start = next(i for i, item in enumerate(body)
                             if isinstance(item, ast.Assign) and ast.unparse(item) == 'optimizer_dirty = True')
                end = next(i for i in range(start + 1, len(body)) if isinstance(body[i], ast.If))
                statements = body[start:end]
                break
        else:
            self.fail('optimizer commit block not found')
        code = compile(ast.Module(body=statements, type_ignores=[]), str(SOURCE), 'exec')
        lines = sorted({item.lineno for item in statements})
        for interrupt_line in lines + [None]:
            with self.subTest(interrupt_line=interrupt_line):
                updates = []
                optimizer = SimpleNamespace(step=lambda: updates.append('applied'), zero_grad=lambda **kwargs: None)
                state = dict(optimizer=optimizer, optimizer_dirty=False, step=800, pending=4,
                             group_index=3203, next_group=3200,
                             safe_payload={'step': 800, 'next_group': 3200}, safe_rng='before')
                state['boundary_payload'] = lambda cursor: dict(step=state['step'], next_group=cursor)
                state['capture_rng_state'] = lambda: 'after'

                def interrupt(frame, event, arg):
                    if event == 'line' and frame.f_code.co_filename == str(SOURCE) and frame.f_lineno == interrupt_line:
                        raise KeyboardInterrupt('injected signal')
                    return interrupt

                sys.settrace(interrupt)
                try:
                    exec(code, state)
                except KeyboardInterrupt:
                    pass
                finally:
                    sys.settrace(None)
                if updates and not state['optimizer_dirty']:
                    self.assertEqual(state['step'], 801)
                    self.assertEqual(state['safe_payload'], {'step': 801, 'next_group': 3204})
                    self.assertEqual(state['safe_rng'], 'after')

    def test_initial_policy_digest_survives_payload_restore(self):
        """Run the real payload writer/restorer with only the heavy model boundary stubbed."""
        tree = trainer_tree()
        functions = {n.name: copy.deepcopy(n) for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name in ('boundary_payload', 'apply_payload')}
        for node in functions.values():
            for statement in ast.walk(node):
                if isinstance(statement, ast.Nonlocal):
                    statement.__class__ = ast.Global
        from recovery_runtime import restore_objects
        ns = dict(copy=copy, step=800, next_group=3200, pending=0, best_key=(.4,.5), best_step=800,
                  active=2, total_tokens=3, prompt_token_cost=4, sampling_teacher_max_difference=0.,
                  inherited_groups=3200, early_stopping=SimpleNamespace(last_step=800),
                  training_diagnostics=SimpleNamespace(samples=3), probability_guards={},
                  log_offsets=lambda path: {}, args=SimpleNamespace(output='unused'),
                  frozen_reference='reference', reference_digest=lambda model:'reference', model=None,
                  initial_policy_parameter_sha256='a'*64, restore_objects=restore_objects,
                  ARITHMETIC_CONFIGURATION={'protocol':'fp32-body-bf16-storage-v1'})
        exec(compile(ast.Module(body=list(functions.values()), type_ignores=[]), str(SOURCE), 'exec'), ns)
        payload = ns['boundary_payload'](3200)
        self.assertEqual(payload.get('initial_policy_parameter_sha256'), 'a'*64)
        ns['initial_policy_parameter_sha256'] = 'b'*64
        ns['apply_payload'](payload)
        self.assertEqual(ns['initial_policy_parameter_sha256'], 'a'*64)


if __name__ == '__main__':
    unittest.main()
