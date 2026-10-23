"""CPU probes of the trainer's actual resource preparation and transfer seam."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).with_name('train_grpo_recoverable.py')


class TrainerWaitPolicyTests(unittest.TestCase):
    def main_body(self):
        tree = ast.parse(SOURCE.read_text())
        return next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')

    def run_statements(self, statements, namespace):
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(SOURCE), 'exec'), namespace)

    def test_source_preparation_does_not_start_memory_reservation(self):
        body = self.main_body()
        end = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                   and ast.unparse(n.targets[0]) == 'source_metrics')
        start = next(i for i,n in enumerate(body[:end]) if isinstance(n,ast.If)
                     and "'TMUX'" in ast.unparse(n.test)) + 1
        def forbid(*args, **kwargs):
            self.fail('Source preparation must not preallocate GPU memory')
        self.run_statements(body[start:end], dict(BackgroundReservation=forbid,
            args=SimpleNamespace(gpu_target_mib=21000), print=lambda *args,**kwargs:None))

    def test_model_transfer_waits_at_idle_gate_with_no_reservation(self):
        body = self.main_body()
        start = next(i for i,n in enumerate(body) if isinstance(n,ast.If)
                     and ast.unparse(n.test) == 'warm and args.resume_checkpoint is None') + 1
        # There are two warm-start blocks; select the model import directly preceding transfer.
        start = max(i for i,n in enumerate(body) if isinstance(n,ast.If)
                    and ast.unparse(n.test) == 'warm and args.resume_checkpoint is None'
                    and any(isinstance(x,ast.Expr) and 'import_policy' in ast.unparse(x) for x in n.body)) + 1
        end = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                   and ast.unparse(n.targets[0]) == 'model.config.use_cache')
        events = []
        def gate(move,rollback,**kwargs):
            events.append(('gate',kwargs['target_mib']))
            # A busy GPU must not receive the model here.
        def forbid(*args,**kwargs):
            self.fail('Trainer still uses progressive reservation')
        model=SimpleNamespace(to=lambda device: events.append(('transfer',device)))
        self.run_statements(body[start:end],dict(
            torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:events.append('empty_cache'))),
            preloader=SimpleNamespace(release=forbid),wait_for_training_gpu=forbid,
            wait_for_idle_training_gpu=gate,model=model,args=SimpleNamespace(gpu_target_mib=21000,output=Path('/unused')),
            write_json=lambda *args:None))
        self.assertEqual(events,['empty_cache',('gate',21000)])


if __name__=='__main__':unittest.main()
