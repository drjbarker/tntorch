import numpy as np
import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)


def test_orthogonalization():

    for i in range(100):
        gt = tn.rand(np.random.randint(1, 8, np.random.randint(2, 6)))
        t = gt.clone()
        assert tn.relative_error(gt, t) <= 1e-7
        t.left_orthogonalize(0)
        assert tn.relative_error(gt, t) <= 1e-7
        t.right_orthogonalize(t.dim()-1)
        assert tn.relative_error(gt, t) <= 1e-7
        t.orthogonalize(np.random.randint(t.dim()))
        assert tn.relative_error(gt, t) <= 1e-7


def test_truncated_svd():
    gt = torch.rand((2, 32, 32))
    u, v = tn.truncated_svd(gt, batch=True)

    for i in range(len(gt)):
        u1, v1 = tn.truncated_svd(gt[i], batch=False)
        assert torch.allclose(u1, u[i])
        assert torch.allclose(v1, v[i])


def test_truncated_svd_eig():
    gt = torch.rand((2, 32, 32))
    u, v = tn.truncated_svd(gt, batch=True, algorithm='eig')

    for i in range(len(gt)):
        u1, v1 = tn.truncated_svd(gt[i], batch=False, algorithm='eig')
        assert torch.allclose(u1, u[i])
        assert torch.allclose(v1, v[i])


def test_truncated_svd_binary_tensor():
    gt = torch.rand((2, 2, 2, 2))
    tt = tn.truncated_svd(gt, eps=1e-12, algorithm='svd')

    assert isinstance(tt, tn.Tensor)
    assert torch.allclose(tt.torch(), gt, atol=1e-10, rtol=1e-10)


def test_truncated_svd_binary_tensor_batch():
    gt = torch.rand((3, 2, 2, 2, 2))
    tt = tn.truncated_svd(gt, rmax=4, algorithm='svd', batch=True)

    assert isinstance(tt, tn.Tensor)
    assert torch.allclose(tt.torch(), gt, atol=1e-10, rtol=1e-10)

    for i in range(len(gt)):
        single = tn.truncated_svd(gt[i], rmax=4, algorithm='svd')
        for j, core in enumerate(single.cores):
            assert torch.allclose(core, tt.cores[j][i, ...])


def test_tensor_binary_svd_init():
    gt = torch.rand((2, 2, 2, 2))
    tt = tn.Tensor(gt, eps=1e-12, algorithm='svd')

    assert torch.allclose(tt.torch(), gt, atol=1e-10, rtol=1e-10)


def test_truncated_svd_binary_zero_tensor_has_minimal_ranks():
    gt = torch.zeros((2, 2, 2, 2))
    tt = tn.truncated_svd(gt, eps=0, algorithm='svd')

    assert isinstance(tt, tn.Tensor)
    assert all(core.shape[0] == 1 for core in tt.cores)
    assert all(core.shape[-1] == 1 for core in tt.cores)
    assert torch.allclose(tt.torch(), gt)


def test_round_tt_svd():

    for i in range(100):
        gt = tn.rand(np.random.randint(1, 8, np.random.randint(8, 10)), ranks_tt=np.random.randint(1, 10))
        gt.round_tt(1e-8, algorithm='svd')
        t = gt+gt
        t.round_tt(1e-8, algorithm='svd')
        assert tn.relative_error(gt, t/2) <= 1e-4
        assert max(gt.ranks_tt) == max(t.ranks_tt)


def test_round_tt_eig():

    for i in range(100):
        gt = tn.rand(np.random.randint(1, 8, np.random.randint(8, 10)), ranks_tt=np.random.randint(1, 10))
        gt.round_tt(1e-8, algorithm='eig')
        t = gt+gt
        t.round_tt(1e-8, algorithm='eig')
        assert tn.relative_error(gt, t/2) <= 1e-7


def test_round_tucker():
        for i in range(100):
            eps = np.random.rand()**2
            gt = tn.rand([32]*4, ranks_tt=8, ranks_tucker=8)
            t = gt.clone()
            t.round_tucker(eps=eps)
            assert tn.relative_error(gt, t) <= eps
