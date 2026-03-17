import pytest
import torch

import tntorch as tn

torch.set_default_dtype(torch.float64)


def _interleaved_dense_matrix(matrix, base=2):
    size = matrix.shape[0]
    num_bits = 0
    value = size
    while value > 1:
        assert value % base == 0
        value //= base
        num_bits += 1
    reshaped = matrix.reshape([base] * (2 * num_bits))
    perm = [i for pair in zip(range(num_bits), range(num_bits, 2 * num_bits)) for i in pair]
    return reshaped.permute(perm)


def test_qtt_from_vector_roundtrip():
    vector = torch.arange(8, dtype=torch.float64).to(torch.cdouble)

    qtt = tn.qtt.from_vector(vector, eps=1e-12)

    assert qtt.shape == torch.Size([2, 2, 2])
    assert tn.qtt.is_qtt(qtt)
    assert not tn.qtt.is_interleaved_operator(qtt)
    assert tn.qtt.vector_length(qtt) == 8
    torch.testing.assert_close(qtt.torch(), vector.reshape(2, 2, 2))


def test_qtt_from_vector_eps_is_relative():
    vector = torch.arange(64, dtype=torch.float64).to(torch.cdouble)
    scaled = 1000 * vector

    qtt = tn.qtt.from_vector(vector, eps=1e-4)
    scaled_qtt = tn.qtt.from_vector(scaled, eps=1e-4)

    torch.testing.assert_close(qtt.ranks_tt, scaled_qtt.ranks_tt)
    assert tn.relative_error(vector.reshape(2, 2, 2, 2, 2, 2), qtt) <= 1e-4
    assert tn.relative_error(scaled.reshape(2, 2, 2, 2, 2, 2), scaled_qtt) <= 1e-4


def test_qtt_from_vector_rejects_non_power_length():
    vector = torch.arange(6, dtype=torch.float64)

    with pytest.raises(ValueError, match="exact power"):
        tn.qtt.from_vector(vector, eps=1e-12)


def test_qtt_from_matrix_interleaved_roundtrip_and_shape():
    matrix = torch.arange(64, dtype=torch.float64).reshape(8, 8).to(torch.cdouble)

    qtt = tn.qtt.from_matrix_interleaved(matrix, eps=1e-12)

    assert qtt.shape == torch.Size([2, 2, 2, 2, 2, 2])
    assert tn.qtt.is_qtt(qtt)
    assert tn.qtt.is_interleaved_operator(qtt)
    assert tn.qtt.matrix_shape(qtt) == (8, 8)
    torch.testing.assert_close(qtt.torch(), _interleaved_dense_matrix(matrix))


def test_qtt_fuse_operator_bits_matches_grouped_dense_tensor():
    matrix = torch.arange(64, dtype=torch.float64).reshape(8, 8).to(torch.cdouble)
    qtt = tn.qtt.from_matrix_interleaved(matrix, eps=1e-12).tt()

    fused = tn.qtt.fuse_operator_bits(qtt)

    expected = _interleaved_dense_matrix(matrix).reshape(4, 4, 4)
    assert fused.shape == torch.Size([4, 4, 4])
    torch.testing.assert_close(fused.torch(), expected)


def test_qtt_from_function_vector_matches_dense_values():
    qtt = tn.qtt.from_function(
        function=lambda bits: torch.sum(bits, dim=1).to(torch.cdouble),
        num_bits=3,
        kind="vector",
        base=2,
        eps=1e-10,
        rmax=4,
        max_iter=10,
        verbose=False,
        suppress_warnings=True,
        detach_evaluations=True,
    )

    coords = torch.cartesian_prod(*([torch.tensor([0.0, 1.0])] * 3))
    expected = torch.sum(coords, dim=1).reshape(2, 2, 2).to(torch.cdouble)
    assert tn.qtt.is_qtt(qtt)
    torch.testing.assert_close(qtt.torch(), expected, atol=1e-8, rtol=1e-8)


def test_qtt_layout_checks_distinguish_vector_from_operator():
    vector = tn.qtt.from_vector(torch.ones(8, dtype=torch.cdouble), eps=1e-12)
    matrix = tn.qtt.from_matrix_interleaved(
        torch.eye(8, dtype=torch.cdouble), eps=1e-12
    )

    assert tn.qtt.is_qtt(vector)
    assert not tn.qtt.is_interleaved_operator(vector)
    assert tn.qtt.is_interleaved_operator(matrix)


def test_quantics_bits_to_int_uses_msb_first_order():
    bits = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=torch.int64,
    )

    values = tn.qtt.quantics_bits_to_int(bits, int_dtype=torch.int64)

    torch.testing.assert_close(values, torch.tensor([5, 3, 6], dtype=torch.int64))


def test_quantics_bits_to_int_applies_along_last_axis():
    bits = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
        ],
        dtype=torch.int64,
    )

    values = tn.qtt.quantics_bits_to_int(bits, int_dtype=torch.int64)

    expected = torch.tensor([[2, 1], [3, 0]], dtype=torch.int64)
    torch.testing.assert_close(values, expected)
