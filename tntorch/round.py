import time
from typing import Optional

import torch


def _binary_tt_svd(
    tensor: torch.Tensor,
    delta: Optional[float] = 0.0,
    rmax=None,
):
    """
    Specialised TT-SVD for tensors whose physical modes are all of size 2.

    This avoids some of the overhead in the generic dense-to-TT path used for
    binary QTT-style tensors.
    """

    import tntorch as tn

    if tensor.ndim == 0:
        return tn.Tensor([tensor.reshape(1, 1, 1)])

    if any(mode != 2 for mode in tensor.shape):
        raise ValueError("Binary TT-SVD expects every mode size to be 2")

    if delta is None:
        delta = 0.0

    d = tensor.ndim
    if rmax is None:
        rmax = [torch.iinfo(torch.int32).max] * max(d - 1, 0)
    elif not hasattr(rmax, "__len__"):
        rmax = [rmax] * max(d - 1, 0)
    else:
        rmax = list(rmax)
        if len(rmax) != d - 1:
            raise ValueError("Expected one TT rank bound per interface")

    work = tensor.to(torch.cdouble if tensor.is_complex() else tensor.dtype)
    local_delta = delta / max((d - 1) ** 0.5, 1.0)
    cores = []
    current = work
    r_prev = 1

    for k in range(d - 1):
        current = current.reshape(r_prev * 2, -1)
        U, S, Vh = torch.linalg.svd(current, full_matrices=False)

        s2 = S.abs().square()
        total = s2.sum()
        discarded = total - torch.cumsum(s2, dim=0)
        rank = int((discarded > local_delta**2).sum().item() + 1)

        rank = max(1, min(rank, S.numel(), int(rmax[k])))

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        cores.append(U.reshape(r_prev, 2, rank))
        current = (S[:, None] * Vh).reshape(rank, *([2] * (d - k - 1)))
        r_prev = rank

    cores.append(current.reshape(r_prev, 2, 1))

    if work.dtype != tensor.dtype:
        cores = [core.to(tensor.dtype) for core in cores]

    return tn.Tensor(cores)


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
    Decomposes a matrix M (size (m x n) in two factors U and V (sizes m x r and
    r x n) with bounded error (or given r).

    As a special case, if a non-batch input has more than two dimensions and all
    modes are of size 2, this dispatches to a specialised binary TT-SVD routine
    and returns a :class:`tntorch.Tensor`.

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
    if hasattr(rmax, "__len__"):
        assert all(rank >= 1 for rank in rmax)
    else:
        assert rmax >= 1
    assert algorithm in ("svd", "eig")

    if not batch and M.ndim > 2 and all(mode == 2 for mode in M.shape):
        if algorithm == "svd":
            return _binary_tt_svd(M, delta=delta, rmax=rmax)

        import tntorch as tn

        norm = torch.norm(M).item()
        tt = tn.Tensor(M, eps=0.0 if norm == 0 else delta / norm, algorithm=algorithm)
        tt.round_tt(rmax=rmax, algorithm=algorithm)
        return tt

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
