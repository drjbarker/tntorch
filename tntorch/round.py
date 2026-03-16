import time
from typing import Optional

import torch


def round_tt(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round_tt()`.

    :param t: input :class:`Tensor`
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round_tt(**kwargs)
    return t2


def round_tucker(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round_tucker()`.

    :param t: input :class:`Tensor`
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round_tucker(**kwargs)
    return t2


def round(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round()`.

    :param t: input :class:`Tensor`
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round(**kwargs)
    return t2


def truncated_svd(
    M: torch.tensor,
    delta: Optional[float] = None,
    eps: Optional[float] = None,
    rmax: Optional[int] = None,
    left_ortho: Optional[bool] = True,
    algorithm: Optional[str] = "svd",
    verbose: Optional[bool] = False,
    batch: Optional[bool] = False,
):
    """
    Decomposes a matrix M (size (m x n) in two factors U and V (sizes m x r and r x n) with bounded error (or given r).

    :param M: a matrix
    :param delta: if provided, maximum error norm
    :param eps: if provided, maximum relative error
    :param rmax: optionally, maximum r
    :param left_ortho: if True (default), U will be orthonormal. If False, V will
    :param algorithm: 'svd' (default) or 'eig'. The latter is often faster, but less accurate
    :param verbose: Boolean
    :param batch: Boolean

    :return: U, V
    """

    if delta is not None and eps is not None:
        raise ValueError("Provide either `delta` or `eps`")
    if delta is None and eps is not None:
        delta = eps * torch.norm(M).item()
    if delta is None and eps is None:
        delta = 0
    if rmax is None:
        rmax = torch.iinfo(torch.int32).max
    assert rmax >= 1
    assert algorithm in ("svd", "eig")

    if batch:
        batch_size = M.shape[0]
        row_dim = M.shape[1]
        col_dim = M.shape[2]
    else:
        row_dim = M.shape[0]
        col_dim = M.shape[1]

    if algorithm == "svd":
        start = time.time()
        left, singular_values, right = torch.linalg.svd(M, full_matrices=False)
        if verbose:
            print("Time (SVD):", time.time() - start)
    else:
        start = time.time()

        if row_dim <= col_dim:
            gram = M @ M.adjoint()
            vector_side = "left"
        else:
            gram = M.adjoint() @ M
            vector_side = "right"

        if verbose:
            print("Time (gram):", time.time() - start)

        start = time.time()
        eigenvalues, vectors = torch.linalg.eigh(gram)
        if verbose:
            print("Time (symmetric EIG):", time.time() - start)

        eigenvalues = torch.clamp(eigenvalues, min=0)
        singular_values = torch.sqrt(eigenvalues)
        if batch:
            reverse = torch.arange(
                singular_values.shape[-1] - 1, -1, -1, device=singular_values.device
            )
            idx = torch.argsort(singular_values, dim=-1)[:, reverse]
            vectors = torch.cat(
                [vectors[i, :, idx[i]][None, ...] for i in range(len(idx))]
            )
            singular_values = torch.cat(
                [singular_values[i, idx[i]][None, ...] for i in range(len(idx))]
            )
        else:
            reverse = torch.arange(
                singular_values.shape[-1] - 1, -1, -1, device=singular_values.device
            )
            idx = torch.argsort(singular_values, dim=-1)[reverse]
            vectors = vectors[..., idx]
            singular_values = singular_values[..., idx]

        safe_inv = torch.where(
            singular_values > 1e-13,
            singular_values.reciprocal(),
            torch.zeros_like(singular_values),
        )
        if vector_side == "left":
            left = vectors
            right = safe_inv[..., :, None] * (left.adjoint() @ M)
        else:
            right = vectors.adjoint()
            left = M @ (vectors * safe_inv[..., None, :])

    # NOTE: Special case: M = zero -> rank is 1
    if batch:
        if singular_values.max() < 1e-13:
            return torch.zeros(
                [batch_size, row_dim, 1], dtype=M.dtype, device=M.device
            ), torch.zeros(
                [batch_size, 1, col_dim], dtype=M.dtype, device=M.device
            )
    else:
        if singular_values[0] < 1e-13:
            return torch.zeros([row_dim, 1], dtype=M.dtype, device=M.device), torch.zeros(
                [1, col_dim], dtype=M.dtype, device=M.device
            )

    S = singular_values**2

    if batch:
        rank = max(1, int(min(rmax, S.shape[-1])))
    else:
        reverse = torch.arange(len(S) - 1, -1, -1)
        where = torch.where((torch.cumsum(S[reverse], dim=0) <= delta**2))[0]

        if len(where) == 0:
            rank = max(1, int(min(rmax, len(S))))
        else:
            rank = max(1, int(min(rmax, len(S) - 1 - where[-1])))

    left = left[..., :rank]
    singular_values = singular_values[..., :rank]
    right = right[..., :rank, :]

    start = time.time()
    if left_ortho:
        M2 = singular_values[..., :, None] * right
    else:
        left = left * singular_values[..., None, :]
        M2 = right

    if verbose:
        print("Time (product):", time.time() - start)

    return left, M2
