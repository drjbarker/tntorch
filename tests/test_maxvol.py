import numpy as np
import torch

from tntorch.maxvol import py_maxvol, py_rect_maxvol, torch_maxvol, torch_rect_maxvol


def _full_rank_matrix(seed, rows, cols, complex_dtype=False):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, cols))
    if complex_dtype:
        matrix = matrix + 1j * rng.normal(size=(rows, cols))
    matrix[:cols] += 2 * np.eye(cols)
    return matrix


def test_py_maxvol_reconstructs_matrix_and_respects_tol():
    A = _full_rank_matrix(seed=0, rows=10, cols=4)
    top_k_index = 6
    tol = 1.0

    index, C = py_maxvol(A, tol=tol, top_k_index=top_k_index)

    assert len(index) == A.shape[1]
    assert np.unique(index).size == len(index)
    assert np.all(index < top_k_index)
    np.testing.assert_allclose(C @ A[index], A, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(C[index], np.eye(A.shape[1]), atol=1e-12, rtol=1e-12)
    assert np.abs(C[:top_k_index]).max() <= tol + 1e-12


def test_py_rect_maxvol_one_step_matches_volume_growth_formula():
    A = _full_rank_matrix(seed=4, rows=7, cols=3, complex_dtype=True)

    index0, C0 = py_maxvol(A, tol=1.05, max_iters=10)
    index, C = py_rect_maxvol(
        A,
        tol=0.0,
        minK=A.shape[1] + 1,
        maxK=A.shape[1] + 1,
        identity_submatrix=False,
        start_maxvol_iters=10,
    )

    added_row = index[-1]
    gram_ratio = np.linalg.det(A[index].conj().T @ A[index]) / np.linalg.det(
        A[index0].conj().T @ A[index0]
    )
    expected_ratio = 1 + np.linalg.norm(C0[added_row]) ** 2

    np.testing.assert_array_equal(index[: A.shape[1]], index0)
    np.testing.assert_allclose(gram_ratio.real, expected_ratio, atol=1e-12, rtol=1e-12)
    assert abs(gram_ratio.imag) <= 1e-12
    np.testing.assert_allclose(C @ A[index], A, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(C, A @ np.linalg.pinv(A[index]), atol=1e-12, rtol=1e-12)


def test_py_rect_maxvol_identity_rows_and_stopping_tolerance():
    A = _full_rank_matrix(seed=2, rows=9, cols=3)

    index, C = py_rect_maxvol(A, tol=1.0, identity_submatrix=True)
    unselected_mask = np.ones(A.shape[0], dtype=bool)
    unselected_mask[index] = False

    np.testing.assert_allclose(C @ A[index], A, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(C[index], np.eye(len(index)), atol=1e-12, rtol=1e-12)
    assert np.linalg.norm(C[unselected_mask], axis=1).max(initial=0.0) <= 1.0 + 1e-12


def test_py_rect_maxvol_min_add_k_forces_extra_rows():
    A = _full_rank_matrix(seed=3, rows=10, cols=4)
    top_k_index = 6

    index, C = py_rect_maxvol(A, tol=10.0, min_add_K=2, top_k_index=top_k_index)

    assert len(index) == A.shape[1] + 2
    assert np.unique(index).size == len(index)
    assert np.all(index < top_k_index)
    np.testing.assert_allclose(C @ A[index], A, atol=1e-11, rtol=1e-11)


def test_torch_maxvol_matches_py_maxvol_real_case():
    A = _full_rank_matrix(seed=7, rows=20, cols=5)

    index_np, C_np = py_maxvol(A, tol=1.05, max_iters=100)
    index_torch, C_torch = torch_maxvol(torch.tensor(A), tol=1.05, max_iters=100)

    np.testing.assert_array_equal(index_torch.cpu().numpy(), index_np)
    np.testing.assert_allclose(C_torch.cpu().numpy(), C_np, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(C_torch.cpu().numpy() @ A[index_np], A, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        C_torch[index_torch].cpu().numpy(), np.eye(A.shape[1]), atol=1e-12, rtol=1e-12
    )


def test_torch_maxvol_matches_py_maxvol_complex_case():
    A = _full_rank_matrix(seed=8, rows=18, cols=4, complex_dtype=True)

    index_np, C_np = py_maxvol(A, tol=1.05, max_iters=100)
    index_torch, C_torch = torch_maxvol(torch.tensor(A), tol=1.05, max_iters=100)

    np.testing.assert_array_equal(index_torch.cpu().numpy(), index_np)
    np.testing.assert_allclose(C_torch.cpu().numpy(), C_np, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(C_torch.cpu().numpy() @ A[index_np], A, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        C_torch[index_torch].cpu().numpy(), np.eye(A.shape[1]), atol=1e-12, rtol=1e-12
    )


def test_torch_rect_maxvol_matches_py_rect_maxvol_real_case():
    A = _full_rank_matrix(seed=9, rows=18, cols=4)

    index_np, C_np = py_rect_maxvol(A, tol=1.0, min_add_K=2, identity_submatrix=False)
    index_torch, C_torch = torch_rect_maxvol(
        torch.tensor(A), tol=1.0, min_add_K=2, identity_submatrix=False
    )

    np.testing.assert_array_equal(index_torch.cpu().numpy(), index_np)
    np.testing.assert_allclose(C_torch.cpu().numpy(), C_np, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        C_torch.cpu().numpy() @ A[index_np], A, atol=1e-11, rtol=1e-11
    )


def test_torch_rect_maxvol_matches_py_rect_maxvol_complex_case():
    A = _full_rank_matrix(seed=10, rows=16, cols=3, complex_dtype=True)

    index_np, C_np = py_rect_maxvol(A, tol=1.0, min_add_K=2, identity_submatrix=True)
    index_torch, C_torch = torch_rect_maxvol(
        torch.tensor(A), tol=1.0, min_add_K=2, identity_submatrix=True
    )

    np.testing.assert_array_equal(index_torch.cpu().numpy(), index_np)
    np.testing.assert_allclose(C_torch.cpu().numpy(), C_np, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        C_torch.cpu().numpy() @ A[index_np], A, atol=1e-11, rtol=1e-11
    )
    np.testing.assert_allclose(
        C_torch[index_torch].cpu().numpy(),
        np.eye(len(index_np)),
        atol=1e-11,
        rtol=1e-11,
    )
