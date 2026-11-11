"""Tensor-membrane exploration — Jean's holographic tensor network as a membrane.

The diagram
-----------
- Boundary physical qubits (top): 7 qubits.
- Tensor layers (middle): three layers of tensors — 6, then 7, then 6.
- Bulk logical qubits (bottom): the protected information, reached by fan-in.

The interpretation: the bulk is the message, the boundary is the hologram,
the tensors are the lens. A bulk logical qubit is delocalised across the
boundary — that is the error correction. This is the same "holographic" as
the membrane work (information stored in a pattern, recoverable by
overlap), in the quantum error-correction language.

This model
-----------
A small, honest, classical simulation of that structure (stdlib + NumPy
only, seeded, reproducible). It is NOT a real quantum error-correcting
code and NOT a real holographic duality.

Construction
------------
- 7 qubits total. Qubits 0 and 1 are the bulk (2 logical qubits, dim 4).
  Qubits 2..6 are ancillas, initialised to |0>.
- The network is 19 random 2-qubit unitaries (the "tensors"), arranged in
  3 layers of 6, 7, 6 — matching the diagram's layer counts. The pairing
  pattern is our concrete choice (the sketch does not specify exact pairs);
  layer 1 fans out from the bulk, layer 2 mixes across, layer 3 mixes
  near the boundary. Each unitary is drawn from a seeded RNG (QR of a
  complex Gaussian), so the network is reproducible.
- The 19 unitaries compose to a 7-qubit unitary U (128x128). The encoding
  V is U restricted to the subspace where the 5 ancillas are |0>: the
  columns of U for input states |b0 b1 0 0 0 0 0>. V is therefore an
  isometry by construction (V^dagger V = I_4), which we verify numerically.

What is measured (see demo.py)
------------------------------
- Isometry: V^dagger V = I, singular values of V.
- Delocalisation: single-boundary-qubit reduced states for different bulk
  states — trace distance between them (near 0 = no single qubit carries
  the bulk).
- Erasure correction: erase k boundary qubits, measure the trace distance
  between the remaining reduced states for different bulk states
  (near 1 = still distinguishable = recoverable). Find the largest k
  that recovers for every subset.
- Overlap recall (membrane-brain style): store the 4 bulk basis states,
  project noisy probes and random probes onto them, name or refuse via a
  threshold tau. Report recognition vs refusal.
"""

import itertools

import numpy as np

__version__ = "0.1.0"

N_QUBITS = 7
N_BULK = 2
N_BOUNDARY = 7
BULK_DIM = 1 << N_BULK          # 4
BOUNDARY_DIM = 1 << N_QUBITS    # 128

# The wiring: which qubit pairs each layer's unitaries act on.
# Qubits 0,1 = bulk. Qubits 2..6 = ancilla / boundary.
# Layer 1 fans out from the bulk, layer 2 mixes across, layer 3 mixes
# near the boundary. Our concrete choice; the sketch does not fix pairs.
LAYER_PAIRS = [
    # layer 1 (bottom, 6 tensors): fan out from the bulk
    [(0, 2), (1, 3), (0, 4), (1, 5), (2, 6), (0, 3)],
    # layer 2 (middle, 7 tensors): mix across
    [(0, 1), (2, 3), (4, 5), (6, 0), (1, 4), (2, 5), (3, 6)],
    # layer 3 (top, 6 tensors): mix near the boundary
    [(0, 1), (2, 3), (4, 5), (5, 6), (0, 6), (1, 2)],
]

# Input basis-state indices for the bulk states |b0 b1 0 0 0 0 0>,
# qubit 0 = most significant bit. b0 -> bit 6 (64), b1 -> bit 5 (32).
BULK_BASIS_INDICES = [0, 32, 64, 96]


def random_unitary_2q(rng):
    """A random 2-qubit unitary (4x4) from a seeded RNG.

    QR of a complex Gaussian matrix, with the diagonal phases of R
    absorbed so the result is a well-defined Haar-random unitary.
    """
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q * ph


def embed_gate(gate, q1, q2, n=N_QUBITS):
    """Embed a 4x4 two-qubit gate into the full 2^n-dim space.

    gate acts on qubits (q1, q2); qubit 0 is the most significant bit.
    Returns a (2^n, 2^n) unitary.
    """
    dim = 1 << n
    G = np.zeros((dim, dim), dtype=complex)
    b1 = n - 1 - q1
    b2 = n - 1 - q2
    for col in range(dim):
        i1 = (col >> b1) & 1
        i2 = (col >> b2) & 1
        in_idx = (i1 << 1) | i2
        base = col & ~(1 << b1) & ~(1 << b2)
        for out_idx in range(4):
            o1 = (out_idx >> 1) & 1
            o2 = out_idx & 1
            row = base | (o1 << b1) | (o2 << b2)
            G[row, col] = gate[out_idx, in_idx]
    return G


def build_unitary(seed=42):
    """Compose the 19 layer unitaries into the full 7-qubit unitary U.

    Returns (U, gates) where gates is the list of (gate, q1, q2) in
    application order (layer 1 first, i.e. closest to the bulk).
    """
    rng = np.random.default_rng(seed)
    dim = 1 << N_QUBITS
    U = np.eye(dim, dtype=complex)
    gates = []
    for pairs in LAYER_PAIRS:
        for (q1, q2) in pairs:
            g = random_unitary_2q(rng)
            gates.append((g, q1, q2))
            U = embed_gate(g, q1, q2) @ U
    return U, gates


def encoding(seed=42):
    """The encoding isometry V (128x4): bulk states -> boundary states.

    V's columns are U applied to |b0 b1 0 0 0 0 0> for the 4 bulk basis
    states, i.e. the columns of U at BULK_BASIS_INDICES.
    """
    U, gates = build_unitary(seed)
    V = U[:, BULK_BASIS_INDICES].copy()
    return V, U, gates


def boundary_state(V, bulk_index):
    """The boundary state for bulk basis state `bulk_index` (0..3)."""
    return V[:, bulk_index].copy()


def partial_trace(state, keep, n=N_QUBITS):
    """Reduced density matrix of a pure state on the qubits in `keep`.

    state: a (2^n,) vector. keep: list of qubit indices to keep.
    Returns a (2^len(keep), 2^len(keep)) density matrix.
    Qubit q corresponds to tensor axis q (axis 0 = most significant bit).
    """
    psi = state.reshape([2] * n)
    trace_axes = [q for q in range(n) if q not in keep]
    rho = np.tensordot(psi, psi.conj(), axes=(trace_axes, trace_axes))
    d = 1 << len(keep)
    return rho.reshape(d, d)


def trace_distance(rho, sigma):
    """(1/2) ||rho - sigma||_1. 0 = identical, 1 = perfectly distinguishable."""
    w = np.linalg.eigvalsh(rho - sigma)
    return 0.5 * float(np.sum(np.abs(w)))


def von_neumann_entropy(rho):
    """Von Neumann entropy (bits) of a density matrix."""
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


def check_isometry(V):
    """Verify V^dagger V = I and return (max deviation, singular values)."""
    M = V.conj().T @ V
    dev = float(np.max(np.abs(M - np.eye(V.shape[1]))))
    svals = np.linalg.svd(V, compute_uv=False)
    return dev, svals


def delocalisation(V):
    """For each boundary qubit, how distinguishable are the bulk states?

    Returns a dict with, per boundary qubit j:
      - mean pairwise trace distance between rho_j(b) for b != b'
        (near 0 = qubit j carries no information about the bulk)
      - mean von Neumann entropy of rho_j(b) (near 1 bit = maximally mixed)
    Also returns the overall means.
    """
    states = [V[:, b] for b in range(BULK_DIM)]
    per_qubit = []
    for j in range(N_BOUNDARY):
        rhos = [partial_trace(s, [j]) for s in states]
        dists = [
            trace_distance(rhos[a], rhos[b])
            for a, b in itertools.combinations(range(BULK_DIM), 2)
        ]
        ents = [von_neumann_entropy(r) for r in rhos]
        per_qubit.append(
            {
                "qubit": j,
                "mean_trace_distance": float(np.mean(dists)),
                "max_trace_distance": float(np.max(dists)),
                "mean_entropy_bits": float(np.mean(ents)),
            }
        )
    return {
        "per_qubit": per_qubit,
        "overall_mean_trace_distance": float(
            np.mean([p["mean_trace_distance"] for p in per_qubit])
        ),
        "overall_mean_entropy_bits": float(
            np.mean([p["mean_entropy_bits"] for p in per_qubit])
        ),
    }


def erasure_recovery(V, max_k=5):
    """Erase k boundary qubits; can the bulk still be recovered?

    For each k and each subset E of k erased qubits, let S be the rest.
    Compute the mean pairwise trace distance between rho_S(b) for b != b'
    (near 1 = the bulk states stay distinguishable from S alone).
    Returns, per k: the minimum over subsets (worst case), the mean over
    subsets, and the number of subsets.
    """
    states = [V[:, b] for b in range(BULK_DIM)]
    pairs = list(itertools.combinations(range(BULK_DIM), 2))
    out = []
    for k in range(0, max_k + 1):
        subset_means = []
        for E in itertools.combinations(range(N_BOUNDARY), k):
            S = [q for q in range(N_BOUNDARY) if q not in E]
            rhos = [partial_trace(s, S) for s in states]
            dists = [trace_distance(rhos[a], rhos[b]) for a, b in pairs]
            subset_means.append(float(np.mean(dists)))
        out.append(
            {
                "erased_k": k,
                "n_subsets": len(subset_means),
                "min_mean_trace_distance": float(np.min(subset_means)),
                "mean_mean_trace_distance": float(np.mean(subset_means)),
            }
        )
    return out


def random_boundary_state(rng):
    """A Haar-random boundary state (128-dim), normalised."""
    g = (rng.standard_normal(BOUNDARY_DIM) + 1j * rng.standard_normal(BOUNDARY_DIM)) / np.sqrt(2.0)
    return g / np.linalg.norm(g)


def noisy_probe(V, bulk_index, sigma, rng):
    """A stored boundary state plus unit-norm noise of amplitude sigma."""
    psi = V[:, bulk_index].copy()
    g = (rng.standard_normal(BOUNDARY_DIM) + 1j * rng.standard_normal(BOUNDARY_DIM)) / np.sqrt(2.0)
    g = g / np.linalg.norm(g)
    probe = psi + sigma * g
    return probe / np.linalg.norm(probe)


def overlap_recall(V, sigma, tau, n_probes_per_pattern=50, n_random=200, seed=1234):
    """Membrane-brain style overlap recall.

    Stored patterns: the 4 bulk basis states, encoded to the boundary.
    Noisy probes: a stored pattern + noise of amplitude sigma.
    Random probes: Haar-random boundary states.
    Decision: name argmax overlap if max overlap >= tau, else refuse.
    Returns recognition rate (noisy probes correctly named) and refusal
    rate (random probes refused), plus the mean max overlaps.
    """
    rng = np.random.default_rng(seed)
    patterns = [V[:, b] for b in range(BULK_DIM)]
    correct = 0
    total = 0
    max_ov_noisy = []
    for b in range(BULK_DIM):
        for _ in range(n_probes_per_pattern):
            probe = noisy_probe(V, b, sigma, rng)
            ovs = [abs(np.vdot(p, probe)) for p in patterns]
            m = int(np.argmax(ovs))
            max_ov_noisy.append(float(np.max(ovs)))
            if np.max(ovs) >= tau and m == b:
                correct += 1
            total += 1
    refused = 0
    max_ov_rand = []
    for _ in range(n_random):
        probe = random_boundary_state(rng)
        ovs = [abs(np.vdot(p, probe)) for p in patterns]
        max_ov_rand.append(float(np.max(ovs)))
        if np.max(ovs) < tau:
            refused += 1
    return {
        "sigma": sigma,
        "tau": tau,
        "recognition_rate": correct / total,
        "refusal_rate": refused / n_random,
        "mean_max_overlap_noisy": float(np.mean(max_ov_noisy)),
        "mean_max_overlap_random": float(np.mean(max_ov_rand)),
    }
