import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from kernel_support import apply_kernel, FUSION_OPTIONS


def test_native_no_optional_import_or_patch(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert not name.startswith('liger_kernel')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert apply_kernel('native')['name']=='native'


@pytest.mark.parametrize('value',['',None,True,'unknown'])
def test_unknown_kernel_rejected(value):
    with pytest.raises(ValueError):apply_kernel(value)


def test_only_linear_cross_entropy_enabled():
    assert FUSION_OPTIONS==dict(rope=False,rms_norm=False,swiglu=False,cross_entropy=False,
                               fused_linear_cross_entropy=True)


def test_liger_patch_uses_only_fused_loss_and_local_vendor(monkeypatch,tmp_path):
    import types
    import kernel_support
    vendor=tmp_path/'vendor';(vendor/'liger_kernel').mkdir(parents=True)
    monkeypatch.setattr(kernel_support,'VENDOR',vendor)
    monkeypatch.setattr(sys,'path',list(sys.path))
    distribution=types.SimpleNamespace(metadata={'Name':'liger_kernel'},version='0.6.2')
    monkeypatch.setattr(kernel_support.metadata,'distributions',lambda **kw:[distribution])
    module=types.ModuleType('liger_kernel');module.__file__=str(vendor/'liger_kernel/__init__.py')
    transformers=types.ModuleType('liger_kernel.transformers');calls=[]
    transformers.apply_liger_kernel_to_qwen3=lambda **kwargs:calls.append(kwargs)
    monkeypatch.setitem(sys.modules,'liger_kernel',module)
    monkeypatch.setitem(sys.modules,'liger_kernel.transformers',transformers)
    result=apply_kernel('liger')
    assert calls==[FUSION_OPTIONS]
    assert result['version']=='0.6.2'
    assert sys.path[0]==str(vendor)


def test_liger_rejects_version_drift(monkeypatch,tmp_path):
    import types
    import kernel_support
    vendor=tmp_path/'vendor';(vendor/'liger_kernel').mkdir(parents=True)
    monkeypatch.setattr(kernel_support,'VENDOR',vendor)
    distribution=types.SimpleNamespace(metadata={'Name':'liger-kernel'},version='0.6.3')
    monkeypatch.setattr(kernel_support.metadata,'distributions',lambda **kw:[distribution])
    with pytest.raises(ValueError,match='version'):
        apply_kernel('liger')
