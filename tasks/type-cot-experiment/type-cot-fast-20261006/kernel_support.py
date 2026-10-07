"""Opt-in local Liger loss fusion; all inference/model component kernels native."""
from importlib import metadata
from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parent/'vendor'
VERSION = '0.6.2'
FUSION_OPTIONS = dict(rope=False, rms_norm=False, swiglu=False,
                      cross_entropy=False, fused_linear_cross_entropy=True)


def apply_kernel(kernel):
    if kernel not in ('native', 'liger'):
        raise ValueError('Training kernel must be native or liger')
    if kernel == 'native':
        return {'name': 'native', 'version': None, 'fusion_options': {}}
    if not (VENDOR/'liger_kernel').is_dir():
        raise ValueError('Local vendored Liger is missing')
    distributions = list(metadata.distributions(path=[str(VENDOR)]))
    installed = [d for d in distributions if d.metadata['Name'].lower().replace('_', '-') == 'liger-kernel']
    if len(installed) != 1 or installed[0].version != VERSION:
        raise ValueError('Vendored Liger version must be exactly 0.6.2')
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import liger_kernel
    if not Path(liger_kernel.__file__).resolve().is_relative_to(VENDOR.resolve()):
        raise ValueError('Liger imported from outside the isolated vendor')
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3(**FUSION_OPTIONS)
    return {'name': 'liger', 'version': VERSION, 'vendor': str(VENDOR),
            'fusion_options': dict(FUSION_OPTIONS)}
