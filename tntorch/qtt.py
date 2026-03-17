from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from .cross import cross
from .tensor import Tensor


def _validate_base(base: int) -> int:
    if not isinstance(base, int) or base < 2:
        raise ValueError("base must be an integer greater than or equal to 2")
    return base


def _absolute_to_relative_eps(norm, eps: float) -> float:
    norm_value = float(norm.item()) if torch.is_tensor(norm) else float(norm)
    if norm_value == 0.0:
        return 0.0
    return float(eps) / norm_value


def _power_length(size: int, base: int) -> int:
    if size < 1:
        raise ValueError("quantized dimensions must be positive")
    count = 0
    value = int(size)
    while value > 1 and value % base == 0:
        value //= base
        count += 1
    if value != 1:
        raise ValueError(f"size {size} is not an exact power of base {base}")
    return count


def _interleaved_permutation(num_row_bits: int, num_col_bits: int):
    perm = []
    paired = min(num_row_bits, num_col_bits)
    for i in range(paired):
        perm.extend((i, num_row_bits + i))
    perm.extend(range(paired, num_row_bits))
    perm.extend(range(num_row_bits + paired, num_row_bits + num_col_bits))
    return perm

@torch.compile
def quantics_bits_to_int(bits: torch.Tensor, int_dtype=torch.int64) -> torch.Tensor:
    """
    Convert quantized digits to integers using most-significant-bit-first order.

    For binary inputs, `[1, 0, 1]` is interpreted as `5`, not `1 + 0*2 + 1*4`.
    The conversion is applied along the last axis.
    """

    bits = torch.as_tensor(bits)
    if bits.dim() == 0:
        raise ValueError("bits must have at least one dimension")
    num_bits = bits.shape[-1]
    powers = torch.arange(
        num_bits - 1, -1, -1, device=bits.device, dtype=int_dtype
    )
    weights = torch.bitwise_left_shift(
        torch.ones(num_bits, device=bits.device, dtype=int_dtype), powers
    )
    return torch.einsum("...b,b->...", bits.to(int_dtype), weights)


def from_vector(
    vector: torch.Tensor,
    eps: float = 1e-12,
    base: int = 2,
    algorithm: str = "svd",
):
    """
    Quantize a dense vector into TT form.
    """

    base = _validate_base(base)
    vector = torch.as_tensor(vector)
    if vector.dim() != 1:
        raise ValueError("vector must be one-dimensional")
    num_bits = _power_length(vector.shape[0], base)
    quantized = vector.reshape([base] * num_bits)
    return Tensor(
        quantized,
        eps=_absolute_to_relative_eps(torch.linalg.norm(quantized), eps),
        algorithm=algorithm,
    )


def from_matrix_interleaved(
    matrix: torch.Tensor,
    eps: float = 1e-12,
    base: int = 2,
    algorithm: str = "svd",
):
    """
    Quantize a dense matrix into interleaved TT form.
    """

    base = _validate_base(base)
    matrix = torch.as_tensor(matrix)
    if matrix.dim() != 2:
        raise ValueError("matrix must be two-dimensional")

    num_row_bits = _power_length(matrix.shape[0], base)
    num_col_bits = _power_length(matrix.shape[1], base)
    reshaped = matrix.reshape([base] * (num_row_bits + num_col_bits))
    interleaved = reshaped.permute(
        _interleaved_permutation(num_row_bits, num_col_bits)
    )
    return Tensor(
        interleaved,
        eps=_absolute_to_relative_eps(torch.linalg.norm(interleaved), eps),
        algorithm=algorithm,
    )


def from_function(
    function: Callable,
    num_bits: Union[int, Tuple[int, int]],
    *,
    kind: str = "vector",
    base: int = 2,
    domain: Optional[Sequence[torch.Tensor]] = None,
    function_arg: str = "matrix",
    **cross_kwargs,
):
    """
    Build a quantized tensor by cross-approximating a black-box function.
    """

    base = _validate_base(base)
    if kind == "vector":
        if not isinstance(num_bits, int):
            raise ValueError("vector QTT expects num_bits to be an integer")
        total_dims = num_bits
    elif kind == "matrix_interleaved":
        if isinstance(num_bits, int):
            num_row_bits = num_bits
            num_col_bits = num_bits
        else:
            num_row_bits, num_col_bits = num_bits
        total_dims = num_row_bits + num_col_bits
    else:
        raise ValueError("kind must be 'vector' or 'matrix_interleaved'")

    if domain is None:
        binary_domain = torch.arange(base, dtype=torch.float64)
        domain = [binary_domain] * total_dims
    elif len(domain) != total_dims:
        raise ValueError(f"expected {total_dims} domain vectors, got {len(domain)}")

    return cross(
        function=function,
        domain=domain,
        function_arg=function_arg,
        **cross_kwargs,
    )


def is_qtt(tensor: Tensor, base: int = 2) -> bool:
    """
    Check whether every physical mode has the expected quantized size.
    """

    base = _validate_base(base)
    if tensor.batch:
        return False
    return all(dim == base for dim in tensor.shape)


def is_interleaved_operator(tensor: Tensor, base: int = 2) -> bool:
    """
    Check whether a tensor looks like an interleaved QTT operator.
    """

    return is_qtt(tensor, base=base) and tensor.dim() % 2 == 0


def vector_length(tensor: Tensor, base: int = 2) -> int:
    """
    Return the represented dense vector length.
    """

    if not is_qtt(tensor, base=base):
        raise ValueError("tensor is not in quantized vector form")
    return int(torch.tensor(tensor.shape).prod().item())


def matrix_shape(tensor: Tensor, base: int = 2, interleaved: bool = True):
    """
    Return the represented dense matrix shape.
    """

    if not is_qtt(tensor, base=base):
        raise ValueError("tensor is not in quantized form")
    dims = list(tensor.shape)
    if interleaved:
        if len(dims) % 2 != 0:
            raise ValueError("interleaved QTT operators require an even number of modes")
        row_dims = dims[0::2]
        col_dims = dims[1::2]
    else:
        midpoint = len(dims) // 2
        row_dims = dims[:midpoint]
        col_dims = dims[midpoint:]
    rows = int(torch.tensor(row_dims).prod().item())
    cols = int(torch.tensor(col_dims).prod().item())
    return rows, cols


@torch.compile
def fuse_operator_bits(tensor: Tensor, base: int = 2, stable: bool = True):
    """
    Fuse adjacent row/column quantization bits into one local operator mode.
    """

    base = _validate_base(base)
    tt = tensor.tt()
    if tt.batch:
        raise ValueError("batched tensors are not supported")
    if not is_interleaved_operator(tt, base=base):
        raise ValueError("tensor is not an interleaved quantized operator")

    fused = []
    source_cores = tt.cores
    for i in range(0, len(source_cores), 2):
        left = source_cores[i]
        right = source_cores[i + 1]
        if left.shape[1] != base or right.shape[1] != base:
            raise ValueError("operator cores do not match the requested quantized base")
        if left.shape[2] != right.shape[0]:
            raise ValueError("adjacent cores have incompatible TT ranks")

        if stable:
            left_mat = left.reshape(left.shape[0] * base, left.shape[2])
            q, r = torch.linalg.qr(left_mat)
            right_mat = right.reshape(right.shape[0], base * right.shape[2])
            right_tilde = (r @ right_mat).reshape(r.shape[0], base, right.shape[2])
            left_tilde = q.reshape(left.shape[0], base, q.shape[1])
            fused_core = torch.einsum(
                "lpr,rqt->lpqt", left_tilde, right_tilde
            ).reshape(left.shape[0], base * base, right.shape[2])
        else:
            fused_core = torch.einsum("lpr,rqt->lpqt", left, right).reshape(
                left.shape[0], base * base, right.shape[2]
            )
        fused.append(fused_core)
    return Tensor(fused)


__all__ = [
    "from_function",
    "from_matrix_interleaved",
    "from_vector",
    "fuse_operator_bits",
    "is_interleaved_operator",
    "is_qtt",
    "matrix_shape",
    "quantics_bits_to_int",
    "vector_length",
]
