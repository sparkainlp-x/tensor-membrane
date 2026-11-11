"""A REAL erasure decoder for the tensor membrane.

The earlier exploration measured DISTINGUISHABILITY: the trace distance
between bulk states on the surviving qubits after k erasures. That is a
necessary condition for recovery, not a demonstrated decoding procedure.

This module builds an actual decoder: given the surviving boundary qubits
after k erasures, reconstruct the bulk state and measure the recovery
fidelity (the overlap between the recovered bulk state and the true one).

Method: least-squares channel inversion (pseudoinverse)
------------------------------------------------------
The erasure is a linear map (a quantum channel) on the bulk:
    N(rho) = Tr_E[ V rho V^dagger ],
where V is the encoding isometry (128x4), E is the erased set, and the
output is the reduced state on the survivors S. A bulk density matrix is
a 4x4 Hermitian matrix (16 real parameters). The map N is linear, so it
can be written as a matrix A acting on the vectorised 4x4 input:
    vec(N(rho)) = A @ vec(rho).
A is built column by column: column (i,j) is vec(N(|i><j|)), i.e. the
reduced operator of the boundary states |psi_i><psi_j|.

The decoder inverts A with a pseudoinverse (SVD, relative cutoff rcond):
    rho_rec = project_to_density_matrix( A^+ @ vec(rho_S_observed) ).
The projection symmetrises, clips negative eigenvalues to 0, and
renormalises the trace to 1. This is a real procedure: it takes the
surviving boundary state and outputs a bulk state. It is NOT a distance
metric.

Two input regimes are measured:
  - exact: the decoder receives the exact reduced state rho_S. This tests
    STRUCTURAL recoverability (is the map injective? is it well
    conditioned?). With exact input and an injective map, recovery is
    exact up to numerical precision.
  - noisy: the decoder receives rho_S + Hermitian Gaussian noise, a proxy
    for finite-sample tomography error. The noise scale is sigma =
    epsilon / d_S, i.e. a fraction epsilon of the typical element size
    (~1/d_S) of a d_S-dimensional density matrix. This tests PRACTICAL
    recoverability: how the recovery degrades when the reduced state is
    only estimated. The threshold can depend on epsilon, and that
    dependence is reported, not hidden.

Assumptions and limits (see README.md for the full list):
  - The decoder knows V and which qubits were erased (standard for an
    erasure channel: the erasure locations are known).
  - The noise model is a proxy, not a simulation of a specific
    tomography protocol.
  - This is a classical simulation of a small toy (2 bulk qubits), not a
    real quantum error-correcting code (no stabilisers, no syndrome, no
    decoder circuit).
"""

import itertools

import numpy as np

from . import model as tm

BULK_DIM = tm.BULK_DIM  # 4


def reduced_op(psi_i, psi_j, keep, n=tm.N_QUBITS):
    """Partial trace of |psi_i><psi_j| onto the qubits in `keep`.

    Returns the (2^len(keep), 2^len(keep)) reduced operator on `keep`.
    For i == j this equals tensor_membrane.partial_trace(psi_i, keep).
    Qubit q corresponds to tensor axis q (axis 0 = most significant bit).
    """
    A = psi_i.reshape([2] * n)
    B = psi_j.reshape([2] * n)
    erase = [q for q in range(n) if q not in keep]
    R = np.tensordot(A, B.conj(), axes=(erase, erase))
    d = 1 << len(keep)
    return R.reshape(d, d)


def build_channel(V, erased, n=tm.N_QUBITS):
    """The erasure channel as a matrix A (d_S^2 x 16).

    A maps vec(rho_bulk) (row-major, 16-dim) to vec(rho_S) (row-major,
    d_S^2-dim), where rho_S = Tr_E[V rho_bulk V^dagger].
    Returns (A, survivors) where survivors is the list of kept qubits.
    """
    survivors = [q for q in range(n) if q not in erased]
    d_s = 1 << len(survivors)
    states = [V[:, b] for b in range(BULK_DIM)]
    A = np.zeros((d_s * d_s, BULK_DIM * BULK_DIM), dtype=complex)
    for i in range(BULK_DIM):
        for j in range(BULK_DIM):
            M = reduced_op(states[i], states[j], survivors, n=n)
            A[:, i * BULK_DIM + j] = M.reshape(-1)
    return A, survivors


def svd_pinv(A, rcond=1e-12):
    """Pseudoinverse of A via SVD with a relative singular-value cutoff.

    Returns (A_pinv, singular_values, kept_rank). Singular values below
    rcond * s_max are treated as zero (dropped).
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.zeros((A.shape[1], A.shape[0]), dtype=complex), s, 0
    cutoff = rcond * s[0]
    keep = s > cutoff
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    A_pinv = (Vh.conj().T * s_inv) @ U.conj().T
    return A_pinv, s, int(np.sum(keep))


def nearest_density_matrix(X):
    """Project a 4x4 matrix onto the nearest density matrix (approx.).

    Symmetrise (Hermitian), clip negative eigenvalues to 0, renormalise
    the trace to 1. Falls back to the maximally mixed state if the
    clipped spectrum sums to ~0.
    """
    X = (X + X.conj().T) / 2.0
    w, U = np.linalg.eigh(X)
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if total <= 1e-15:
        return np.eye(X.shape[0], dtype=complex) / X.shape[0]
    w = w / total
    return (U * w) @ U.conj().T


def fidelity_pure(psi, rho):
    """Overlap <psi|rho|psi> for a pure state psi and density matrix rho."""
    return float(np.real(np.vdot(psi, rho @ psi)))


def hermitian_gaussian_noise(dim, sigma, rng):
    """A Hermitian Gaussian noise matrix (dim x dim).

    Built as G = (Z + Z^dagger)/2 with Z complex Gaussian of per-entry
    std sigma. Independent real/imaginary parts, so each element of G
    has RMS ~ sigma/sqrt(2). The decoder does NOT know this noise.
    """
    Z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) * sigma
    return (Z + Z.conj().T) / 2.0


class ErasureDecoder:
    """Least-squares decoder for one erasure pattern.

    Knows V and the erased set (standard: erasure locations are known).
    decode(rho_S_observed) returns the recovered 4x4 bulk density matrix.
    """

    def __init__(self, V, erased, rcond=1e-12):
        self.erased = tuple(sorted(erased))
        self.A, self.survivors = build_channel(V, self.erased)
        self.d_s = 1 << len(self.survivors)
        self.A_pinv, self.svals, self.rank = svd_pinv(self.A, rcond=rcond)
        self.s_max = float(self.svals[0]) if self.svals.size else 0.0
        self.s_min_kept = float(self.svals[self.rank - 1]) if self.rank else 0.0
        self.cond = self.s_max / self.s_min_kept if self.s_min_kept > 0 else float("inf")
        self.full_rank = self.rank == BULK_DIM * BULK_DIM

    def decode(self, rho_S_obs):
        """Recover the bulk density matrix from an observed reduced state."""
        y = np.asarray(rho_S_obs, dtype=complex).reshape(-1)
        x = self.A_pinv @ y
        X = x.reshape(BULK_DIM, BULK_DIM)
        return nearest_density_matrix(X)

    def conditioning(self):
        """Singular values and condition number of the channel matrix."""
        return {
            "rank": self.rank,
            "full_rank": self.full_rank,
            "s_max": self.s_max,
            "s_min_kept": self.s_min_kept,
            "cond": self.cond,
            "singular_values": [float(s) for s in self.svals],
        }


def random_bulk_states(n_states, seed=777):
    """Haar-random pure bulk states (4-dim), fixed set for comparability."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_states):
        g = (rng.standard_normal(BULK_DIM) + 1j * rng.standard_normal(BULK_DIM)) / np.sqrt(2.0)
        states.append(g / np.linalg.norm(g))
    return states


def measure_recovery(V, max_k=5, n_random=16, epsilons=(0.0, 0.01, 0.1, 0.5),
                     noise_seed=999, rcond=1e-12):
    """Measure recovery fidelity for each erasure count k.

    For every k and every subset E of k erased qubits:
      - build the decoder, record its conditioning;
      - for each test bulk state (4 basis states + n_random random pure
        states): compute the exact reduced state on the survivors, decode
        it (exact input), and decode a noisy copy (one noise draw per
        state per epsilon).
    Returns a dict with, per k and per epsilon:
      - worst-case (min) and mean fidelity over subsets and states;
      - the worst-case (largest) condition number over subsets;
      - the number of subsets.
    Exact input is epsilon = 0.0.
    """
    # Fixed test states, reused across all (k, subset) for comparability.
    basis = [np.eye(BULK_DIM, dtype=complex)[:, b] for b in range(BULK_DIM)]
    random_states = random_bulk_states(n_random)
    test_states = basis + random_states

    out = {"epsilons": list(epsilons), "per_k": []}
    for k in range(0, max_k + 1):
        per_eps = {eps: {"fidelities": []} for eps in epsilons}
        conds = []
        ranks = []
        n_subsets = 0
        for E in itertools.combinations(range(tm.N_BOUNDARY), k):
            n_subsets += 1
            dec = ErasureDecoder(V, E, rcond=rcond)
            conds.append(dec.cond)
            ranks.append(dec.rank)
            S = dec.survivors
            # One noise RNG per subset, seeded deterministically, so the
            # measurement is reproducible and subsets are comparable.
            rng = np.random.default_rng(noise_seed + 1000 * k + n_subsets)
            for psi in test_states:
                full = V @ psi
                rho_S = tm.partial_trace(full, S)
                for eps in epsilons:
                    if eps == 0.0:
                        rho_obs = rho_S
                    else:
                        sigma = eps / dec.d_s
                        rho_obs = rho_S + hermitian_gaussian_noise(dec.d_s, sigma, rng)
                    rec = dec.decode(rho_obs)
                    per_eps[eps]["fidelities"].append(fidelity_pure(psi, rec))
        row = {
            "erased_k": k,
            "n_subsets": n_subsets,
            "n_states_per_subset": len(test_states),
            "min_rank": int(min(ranks)),
            "max_rank": int(max(ranks)),
            "max_cond": float(max(conds)),
            "mean_cond": float(np.mean(conds)),
            "by_epsilon": {},
        }
        for eps in epsilons:
            fids = per_eps[eps]["fidelities"]
            row["by_epsilon"][str(eps)] = {
                "worst_fidelity": float(np.min(fids)),
                "mean_fidelity": float(np.mean(fids)),
            }
        out["per_k"].append(row)
    return out
