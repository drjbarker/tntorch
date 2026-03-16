import importlib

import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)
from util import random_format


def test_domain():

    def function(Xs):
        return 1. / torch.sum(Xs, dim=1)

    domain = [torch.linspace(1, 10, 10) for n in range(3)]
    t = tn.cross(function=function, domain=domain, ranks_tt=3, function_arg='matrix')
    gt = torch.meshgrid(domain, indexing='ij')
    gt = 1. / sum(gt)

    assert tn.relative_error(gt, t) < 5e-2


def test_tensors():

    for i in range(100):
        t = random_format([10] * 6)
        t2 = tn.cross(function=lambda x: x, tensors=t, ranks_tt=15, verbose=False)
        assert tn.relative_error(t, t2) < 1e-6

    t = tn.rand([10] * 6, ranks_tt=10)
    _, info = tn.cross(function=lambda x: x, tensors=[t], ranks_tt=15, verbose=False, return_info=True)
    t2 = tn.cross_forward(info, function=lambda x: x, tensors=t)
    assert tn.relative_error(t, t2) < 1e-6


def test_ops():

    x, y, z, w = tn.meshgrid([32]*4)
    t = x + y + z + w + 1
    assert tn.relative_error(1/t.torch(), 1/t) < 1e-4
    assert tn.relative_error(torch.cos(t.torch()), tn.cos(t)) < 1e-4
    assert tn.relative_error(torch.exp(t.torch()), tn.exp(t)) < 1e-4


def test_cross_uses_py_maxvol_on_cpu(monkeypatch):
    cross_module = importlib.import_module("tntorch.cross")
    calls = {"py_maxvol": 0, "torch_maxvol": 0}
    original_py = cross_module.py_maxvol
    original_torch = cross_module.torch_maxvol

    def wrapped_py(*args, **kwargs):
        calls["py_maxvol"] += 1
        return original_py(*args, **kwargs)

    def wrapped_torch(*args, **kwargs):
        calls["torch_maxvol"] += 1
        return original_torch(*args, **kwargs)

    monkeypatch.setattr(cross_module, "py_maxvol", wrapped_py)
    monkeypatch.setattr(cross_module, "torch_maxvol", wrapped_torch)
    t = tn.rand([6, 6, 6], ranks_tt=3)
    tn.cross(function=lambda x: x, tensors=t, ranks_tt=3, verbose=False)

    assert calls["py_maxvol"] > 0
    assert calls["torch_maxvol"] == 0


def test_cross_minimise_uses_py_rect_maxvol_on_cpu(monkeypatch):
    cross_module = importlib.import_module("tntorch.cross")
    calls = {"py_rect_maxvol": 0, "torch_rect_maxvol": 0}
    original_py = cross_module.py_rect_maxvol
    original_torch = cross_module.torch_rect_maxvol

    def wrapped_py(*args, **kwargs):
        calls["py_rect_maxvol"] += 1
        return original_py(*args, **kwargs)

    def wrapped_torch(*args, **kwargs):
        calls["torch_rect_maxvol"] += 1
        return original_torch(*args, **kwargs)

    monkeypatch.setattr(cross_module, "py_rect_maxvol", wrapped_py)
    monkeypatch.setattr(cross_module, "torch_rect_maxvol", wrapped_torch)
    domain = [torch.linspace(1, 2, 4) for _ in range(3)]
    tn.cross(
        function=lambda x, y, z: x + y + z,
        domain=domain,
        ranks_tt=2,
        verbose=False,
        _minimize=True,
    )

    assert calls["py_rect_maxvol"] > 0
    assert calls["torch_rect_maxvol"] == 0
