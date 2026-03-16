import pytest
import torch

import tntorch as tn

torch.set_default_dtype(torch.float64)


def _real_dtype(dtype):
    """Map a complex dtype to its corresponding real dtype."""
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise ValueError("Unsupported complex dtype: {}".format(dtype))


def _random_complex(shape, dtype):
    """Generate a complex tensor with independent normal real and imaginary parts."""
    real_dtype = _real_dtype(dtype)
    real = torch.randn(shape, dtype=real_dtype)
    imag = torch.randn(shape, dtype=real_dtype)
    return torch.complex(real, imag).to(dtype)


def _hermitian_inner(x, y):
    """Compute the dense Hermitian inner product used as the test oracle."""
    return torch.sum(x.conj() * y)


def _tol(dtype, *, loose=False):
    """Return tolerances tuned for complex64 versus complex128 test comparisons."""
    if dtype == torch.complex64:
        if loose:
            return {"rtol": 5e-3, "atol": 5e-4}
        return {"rtol": 5e-4, "atol": 5e-5}
    if loose:
        return {"rtol": 1e-10, "atol": 1e-10}
    return {"rtol": 1e-12, "atol": 1e-12}


def _assert_close(actual, expected, dtype, *, loose=False):
    """Assert two tensors are close using dtype-specific tolerances."""
    torch.testing.assert_close(actual, expected, **_tol(dtype, loose=loose))


def _assert_hermitian_columns(left, dtype):
    """Check that a matrix has approximately orthonormal columns in the Hermitian sense."""
    gram = left.conj().transpose(-1, -2) @ left
    eye = torch.eye(left.shape[-1], dtype=dtype, device=left.device)
    _assert_close(gram, eye, dtype, loose=True)


def _assert_hermitian_rows(right, dtype):
    """Check that a matrix has approximately orthonormal rows in the Hermitian sense."""
    gram = right @ right.conj().transpose(-1, -2)
    eye = torch.eye(right.shape[0], dtype=dtype, device=right.device)
    _assert_close(gram, eye, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_tensor_roundtrip(dtype):
    """Dense complex tensors should round-trip through tn.Tensor without losing dtype or values."""
    x = _random_complex((4, 3, 5), dtype)
    t = tn.Tensor(x)

    assert t.dtype == dtype
    assert all(core.dtype == dtype for core in t.cores)
    _assert_close(t.torch(), x, dtype)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_batch_tensor_roundtrip(dtype):
    """Batched dense complex tensors should round-trip through tn.Tensor unchanged."""
    x = _random_complex((2, 4, 3, 5), dtype)
    t = tn.Tensor(x, batch=True)

    assert t.dtype == dtype
    assert all(core.dtype == dtype for core in t.cores)
    _assert_close(t.torch(), x, dtype)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("algorithm", ["svd", "eig"])
def test_complex_tt_reconstruction(dtype, algorithm):
    """Exact-rank TT construction should reconstruct a dense complex tensor."""
    x = _random_complex((3, 4, 5), dtype)
    t = tn.Tensor(x, ranks_tt=[3, 5], algorithm=algorithm)

    _assert_close(t.torch(), x, dtype, loose=(algorithm == "eig"))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("algorithm", ["svd", "eig"])
def test_complex_tucker_reconstruction(dtype, algorithm):
    """Exact-rank Tucker construction should reconstruct a dense complex tensor."""
    x = _random_complex((3, 4, 5), dtype)
    t = tn.Tensor(x, ranks_tucker=[3, 4, 5], algorithm=algorithm)

    _assert_close(t.torch(), x, dtype, loose=(algorithm == "eig"))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_arithmetic_matches_dense(dtype):
    """Complex tensor arithmetic should match dense PyTorch results."""
    x = _random_complex((3, 4, 2), dtype)
    y = _random_complex((3, 4, 2), dtype)
    alpha = 1.25 - 0.5j

    t1 = tn.Tensor(x, ranks_tt=[3, 2])
    t2 = tn.Tensor(y, ranks_tucker=[3, 4, 2])

    dense1 = t1.torch()
    dense2 = t2.torch()

    _assert_close((t1 + t2).torch(), dense1 + dense2, dtype, loose=True)
    _assert_close((t1 - t2).torch(), dense1 - dense2, dtype, loose=True)
    _assert_close((t1 * t2).torch(), dense1 * dense2, dtype, loose=True)
    _assert_close((alpha * t1).torch(), alpha * dense1, dtype, loose=True)
    _assert_close((t1 * alpha).torch(), dense1 * alpha, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_metrics_use_hermitian_inner_product(dtype):
    """Metric helpers should agree with dense Hermitian inner-product semantics."""
    x = _random_complex((3, 4, 2), dtype)
    y = _random_complex((3, 4, 2), dtype)

    t1 = tn.Tensor(x, ranks_tt=[3, 2])
    t2 = tn.Tensor(y, ranks_tt=[3, 2])

    dense1 = t1.torch()
    dense2 = t2.torch()

    expected_dot = _hermitian_inner(dense1, dense2)
    expected_normsq = _hermitian_inner(dense1, dense1)
    expected_norm = torch.linalg.norm(dense1)
    expected_dist = torch.linalg.norm(dense1 - dense2)
    expected_relative_error = expected_dist / expected_norm

    _assert_close(tn.dot(t1, t2), expected_dot, dtype, loose=True)
    _assert_close(tn.dot(t1, t1), expected_normsq, dtype, loose=True)
    _assert_close(tn.normsq(t1), expected_normsq, dtype, loose=True)
    _assert_close(tn.norm(t1), expected_norm, dtype, loose=True)
    _assert_close(tn.dist(t1, t2), expected_dist, dtype, loose=True)
    _assert_close(
        tn.relative_error(t1, t2), expected_relative_error, dtype, loose=True
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("algorithm", ["svd", "eig"])
@pytest.mark.parametrize("batch", [False, True])
def test_complex_truncated_svd_reconstructs_matrix(dtype, algorithm, batch):
    """Truncated SVD should reconstruct complex matrices and preserve orthogonality."""
    if batch:
        matrix = _random_complex((2, 6, 4), dtype)
    else:
        matrix = _random_complex((6, 4), dtype)

    left, right = tn.truncated_svd(matrix, algorithm=algorithm, batch=batch)

    _assert_close(left @ right, matrix, dtype, loose=(algorithm == "eig"))

    if batch:
        for idx in range(matrix.shape[0]):
            _assert_hermitian_columns(left[idx], dtype)
    else:
        _assert_hermitian_columns(left, dtype)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_orthogonalization_preserves_tensor(dtype):
    """Orthogonalisation steps should preserve the represented complex tensor."""
    x = _random_complex((3, 4, 2), dtype)
    t = tn.Tensor(x, ranks_tt=[3, 2])
    reference = t.torch().clone()

    t.left_orthogonalize(0)
    _assert_close(t.torch(), reference, dtype, loose=True)
    _assert_hermitian_columns(tn.left_unfolding(t.cores[0]), dtype)

    t.right_orthogonalize(t.dim() - 1)
    _assert_close(t.torch(), reference, dtype, loose=True)
    _assert_hermitian_rows(tn.right_unfolding(t.cores[-1]), dtype)

    t.orthogonalize(1)
    _assert_close(t.torch(), reference, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("algorithm", ["svd", "eig"])
def test_complex_round_tt_preserves_tensor(dtype, algorithm):
    """TT rounding should keep a complex tensor unchanged when no truncation is needed."""
    x = _random_complex((3, 4, 2), dtype)
    t = tn.Tensor(x, ranks_tt=[3, 2], algorithm=algorithm)
    reference = t.torch().clone()

    t.round_tt(1e-10, algorithm=algorithm)

    _assert_close(t.torch(), reference, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("algorithm", ["svd", "eig"])
def test_complex_round_tucker_preserves_tensor(dtype, algorithm):
    """Tucker rounding should keep a complex tensor unchanged when no truncation is needed."""
    x = _random_complex((3, 4, 2), dtype)
    t = tn.Tensor(x, ranks_tucker=[3, 4, 2], algorithm=algorithm)
    reference = t.torch().clone()

    t.round_tucker(eps=1e-10, algorithm=algorithm)

    _assert_close(t.torch(), reference, dtype, loose=True)
