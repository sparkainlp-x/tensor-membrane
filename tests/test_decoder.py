"""Tests for the real erasure decoder (scratch-decoder).

Run: python -m pytest tests_decoder.py -q
Covers: reduced_op vs the original partial_trace, decoder correctness
with no erasure, recovery fidelity monotone in k, determinism, and the
channel conditioning (rank / singular values).
"""

import numpy as np
import pytest

from tensor_membrane import model as tm
from tensor_membrane import decoder as dec


@pytest.fixture(scope="module")
def V42():
    V, _, _ = tm.encoding(seed=42)
    return V


def test_reduced_op_matches_partial_trace(V42):
    """reduced_op(psi, psi, keep) == partial_trace(psi, keep)."""
    for b in range(tm.BULK_DIM):
        psi = V42[:, b]
        for keep in ([0], [0, 1], [2, 3, 4], [0, 2, 4, 6], list(range(7))):
            a = dec.reduced_op(psi, psi, keep)
            c = tm.partial_trace(psi, keep)
            assert np.allclose(a, c, atol=1e-12), (b, keep)


def test_reduced_op_offdiag_is_trace_one_free():
    """reduced_op(psi_i, psi_j) is traceless for i != j (orthogonal states)."""
    V, _, _ = tm.encoding(seed=42)
    for i, j in [(0, 1), (1, 3), (2, 0)]:
        M = dec.reduced_op(V[:, i], V[:, j], [0, 1, 2])
        assert abs(np.trace(M)) < 1e-12, (i, j, np.trace(M))


def test_channel_matrix_reproduces_partial_trace(V42):
    """A @ vec(|b><b|) == vec(partial_trace(V|b>))."""
    A, S = dec.build_channel(V42, erased=(0,))
    for b in range(tm.BULK_DIM):
        x = np.zeros(16, dtype=complex)
        x[b * 4 + b] = 1.0
        y = A @ x
        rho = tm.partial_trace(V42[:, b], S)
        assert np.allclose(y.reshape(rho.shape), rho, atol=1e-12), b


def test_decoder_no_erasure_is_exact(V42):
    """With no erasure the decoder recovers every test state exactly."""
    d = dec.ErasureDecoder(V42, erased=())
    states = [np.eye(4, dtype=complex)[:, b] for b in range(4)] + dec.random_bulk_states(8)
    for psi in states:
        rho = tm.partial_trace(V42 @ psi, list(range(7)))
        rec = d.decode(rho)
        assert dec.fidelity_pure(psi, rec) > 1 - 1e-10, dec.fidelity_pure(psi, rec)


def test_decoder_output_is_a_density_matrix(V42):
    """decode() always returns Hermitian, trace-1, PSD output."""
    d = dec.ErasureDecoder(V42, erased=(0, 3))
    rng = np.random.default_rng(5)
    for _ in range(5):
        rho = tm.partial_trace(V42 @ dec.random_bulk_states(1, seed=int(rng.integers(1e6)))[0],
                               d.survivors)
        rho = rho + dec.hermitian_gaussian_noise(d.d_s, 0.05, rng)
        rec = d.decode(rho)
        assert np.allclose(rec, rec.conj().T, atol=1e-12)
        assert abs(np.trace(rec) - 1.0) < 1e-12
        assert np.min(np.linalg.eigvalsh(rec)) >= -1e-12


def test_channel_full_rank_up_to_k4(V42):
    """The channel matrix is full rank (16) for k <= 4 on every subset."""
    for k in range(0, 5):
        import itertools
        for E in itertools.combinations(range(7), k):
            d = dec.ErasureDecoder(V42, erased=E)
            assert d.full_rank, (k, E, d.rank)


def test_fidelity_monotone_in_k_exact(V42):
    """Exact-input worst-case fidelity is non-increasing in k."""
    res = dec.measure_recovery(V42, max_k=5, n_random=4, epsilons=(0.0,))
    worst = [row["by_epsilon"]["0.0"]["worst_fidelity"] for row in res["per_k"]]
    for a, b in zip(worst, worst[1:]):
        assert b <= a + 1e-9, (worst,)


def test_measure_recovery_deterministic(V42):
    """Two identical runs give identical results."""
    r1 = dec.measure_recovery(V42, max_k=2, n_random=4, epsilons=(0.0, 0.1))
    r2 = dec.measure_recovery(V42, max_k=2, n_random=4, epsilons=(0.0, 0.1))
    assert r1 == r2


def test_noise_hurts_more_at_larger_k(V42):
    """At fixed noise, worst-case fidelity at k=2 is below that at k=0."""
    res = dec.measure_recovery(V42, max_k=2, n_random=8, epsilons=(0.1,))
    w0 = res["per_k"][0]["by_epsilon"]["0.1"]["worst_fidelity"]
    w2 = res["per_k"][2]["by_epsilon"]["0.1"]["worst_fidelity"]
    assert w2 < w0, (w0, w2)


def test_conditioning_grows_with_k(V42):
    """Worst-case condition number grows (not shrinks) with k."""
    res = dec.measure_recovery(V42, max_k=5, n_random=2, epsilons=(0.0,))
    conds = [row["max_cond"] for row in res["per_k"]]
    assert conds[-1] >= conds[0], conds
