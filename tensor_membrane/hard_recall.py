"""Hard overlap recall for the tensor membrane.

The easy version (tensor_membrane.overlap_recall) stores the 4 mutually
orthogonal bulk basis states, so random probes overlap ~0.12 while noisy
probes stay ~0.99 -- the gate is not doing hard work. Here the stored bulk
patterns are NON-ORTHOGONAL with a controlled pairwise overlap, so recall
is genuinely hard. A second, harder probe type is added: a random bulk
state encoded to the boundary (a state in the code subspace that is NOT a
stored pattern). The overlap gate must refuse it -- and, as in
membrane-brain, it may not be able to.

Pattern construction
--------------------
psi_j = (sqrt(1 - s^2) * e_j + s * c) / N,  c = uniform (1,1,1,1)/2,
N = sqrt(1 + s*sqrt(1 - s^2)).  s = 0 gives the orthonormal basis;
s = 1 gives all patterns identical (= c).  The measured mean pairwise
|overlap| rho(s) = (s*sqrt(1-s^2) + s^2) / (1 + s*sqrt(1-s^2)) rises
monotonically from 0 to 1.  Because V is an isometry, the boundary
patterns have exactly the same pairwise overlaps as the bulk patterns.

Probe types
-----------
- noisy: a stored pattern + noise of amplitude sigma (as in the easy test).
- isotropic random: a Haar-random 128-dim boundary state.
- code-random: a Haar-random bulk state (dim 4) encoded to the boundary.
  It lives in the code subspace but is not a stored pattern.  The gate
  should refuse it; whether it can is the hard question.

Decision (unchanged): name argmax overlap if max overlap >= tau, else
refuse.  Recognition counts only correctly named AND accepted probes.
Misnaming (argmax wrong but accepted) and misses (real pattern refused)
are tracked separately.
"""

import itertools

import numpy as np

from . import model as tm


def similar_bulk_patterns(s, n_patterns=4):
    """Build n_patterns bulk states (dim 4) with controlled similarity.

    s in [0, 1].  Returns (patterns, rho) where patterns is a list of
    unit vectors (dim 4) and rho is the measured mean pairwise
    |overlap| between distinct patterns.
    """
    if not 0.0 <= s <= 1.0:
        raise ValueError("s must be in [0, 1]")
    if n_patterns > tm.BULK_DIM:
        raise ValueError("n_patterns cannot exceed the bulk dimension 4")
    a = np.sqrt(1.0 - s * s)
    c = np.full(tm.BULK_DIM, 1.0 / np.sqrt(tm.BULK_DIM), dtype=complex)
    norm = np.sqrt(1.0 + s * a)
    patterns = []
    for j in range(n_patterns):
        e = np.zeros(tm.BULK_DIM, dtype=complex)
        e[j] = 1.0
        psi = (a * e + s * c) / norm
        patterns.append(psi / np.linalg.norm(psi))
    ovs = [
        abs(np.vdot(patterns[i], patterns[j]))
        for i, j in itertools.combinations(range(n_patterns), 2)
    ]
    return patterns, float(np.mean(ovs))


def encode_patterns(V, bulk_patterns):
    """Encode bulk patterns to the boundary: phi_j = V psi_j."""
    return [V @ p for p in bulk_patterns]


def code_random_probe(V, rng):
    """A Haar-random bulk state (dim 4) encoded to the boundary."""
    g = (rng.standard_normal(tm.BULK_DIM) + 1j * rng.standard_normal(tm.BULK_DIM)) / np.sqrt(2.0)
    psi = g / np.linalg.norm(g)
    return V @ psi


def hard_recall(V, s, sigma, tau, n_probes_per_pattern=100, n_random=300,
                n_code_random=300, seed=1234):
    """Hard overlap recall at one (similarity, noise, threshold) point.

    Returns recognition rate, misname rate, miss rate, refusal rates for
    isotropic-random and code-random probes, mean max overlaps, and the
    measured pairwise overlap rho of the stored patterns.
    """
    bulk_patterns, rho = similar_bulk_patterns(s)
    patterns = encode_patterns(V, bulk_patterns)
    rng = np.random.default_rng(seed)

    recognised = 0
    misnamed = 0
    missed = 0
    total = 0
    max_ov_noisy = []
    for b in range(len(patterns)):
        for _ in range(n_probes_per_pattern):
            probe = _noisy_probe_pattern(patterns[b], sigma, rng)
            ovs = [abs(np.vdot(p, probe)) for p in patterns]
            m = int(np.argmax(ovs))
            max_ov_noisy.append(float(np.max(ovs)))
            if np.max(ovs) < tau:
                missed += 1
            elif m == b:
                recognised += 1
            else:
                misnamed += 1
            total += 1

    refused_random = 0
    max_ov_rand = []
    for _ in range(n_random):
        probe = tm.random_boundary_state(rng)
        ovs = [abs(np.vdot(p, probe)) for p in patterns]
        max_ov_rand.append(float(np.max(ovs)))
        if np.max(ovs) < tau:
            refused_random += 1

    refused_code = 0
    max_ov_code = []
    for _ in range(n_code_random):
        probe = code_random_probe(V, rng)
        ovs = [abs(np.vdot(p, probe)) for p in patterns]
        max_ov_code.append(float(np.max(ovs)))
        if np.max(ovs) < tau:
            refused_code += 1

    return {
        "s": s,
        "rho": rho,
        "sigma": sigma,
        "tau": tau,
        "recognition_rate": recognised / total,
        "misname_rate": misnamed / total,
        "miss_rate": missed / total,
        "refusal_rate_random": refused_random / n_random,
        "refusal_rate_code": refused_code / n_code_random,
        "mean_max_overlap_noisy": float(np.mean(max_ov_noisy)),
        "mean_max_overlap_random": float(np.mean(max_ov_rand)),
        "mean_max_overlap_code": float(np.mean(max_ov_code)),
    }


def _noisy_probe_pattern(pattern, sigma, rng):
    """Stored boundary pattern + unit-norm noise of amplitude sigma."""
    g = (rng.standard_normal(tm.BOUNDARY_DIM) + 1j * rng.standard_normal(tm.BOUNDARY_DIM)) / np.sqrt(2.0)
    g = g / np.linalg.norm(g)
    probe = pattern + sigma * g
    return probe / np.linalg.norm(probe)


def subspace_noisy_probe(V, bulk_pattern, sigma, rng):
    """Perturb the BULK state, then encode.

    probe = V . normalize(psi_j + sigma * h), h a random unit vector in
    the 4-dim bulk space.  Unlike isotropic 128-dim boundary noise, this
    moves the state inside the code subspace, so it can flip the argmax
    between similar stored patterns.  This is the genuinely hard noise.
    """
    h = (rng.standard_normal(tm.BULK_DIM) + 1j * rng.standard_normal(tm.BULK_DIM)) / np.sqrt(2.0)
    h = h / np.linalg.norm(h)
    psi = bulk_pattern + sigma * h
    psi = psi / np.linalg.norm(psi)
    return V @ psi


def collision_probe(V, bulk_a, bulk_b, rng):
    """A superposition of two stored bulk patterns, encoded.

    probe = V . normalize(psi_a + psi_b).  A genuine ambiguity: the probe
    is (nearly) equally close to two stored patterns, like a planted
    collision in membrane-arena.  The gate must name one; the overlap
    with each is ~ (1 + rho) / sqrt(2 + 2 rho).
    """
    psi = bulk_a + bulk_b
    psi = psi / np.linalg.norm(psi)
    return V @ psi


def subspace_recall(V, s, sigma, tau, n_probes_per_pattern=100, n_random=300,
                    n_code_random=300, seed=1234):
    """Hard recall with SUBSPACE noise (perturbation inside the code space).

    Same metrics as hard_recall, but noisy probes are built by perturbing
    the bulk state before encoding.  This is the noise that can actually
    confuse similar stored patterns.
    """
    bulk_patterns, rho = similar_bulk_patterns(s)
    patterns = encode_patterns(V, bulk_patterns)
    rng = np.random.default_rng(seed)

    recognised = 0
    misnamed = 0
    missed = 0
    total = 0
    max_ov_noisy = []
    for b in range(len(patterns)):
        for _ in range(n_probes_per_pattern):
            probe = subspace_noisy_probe(V, bulk_patterns[b], sigma, rng)
            ovs = [abs(np.vdot(p, probe)) for p in patterns]
            m = int(np.argmax(ovs))
            max_ov_noisy.append(float(np.max(ovs)))
            if np.max(ovs) < tau:
                missed += 1
            elif m == b:
                recognised += 1
            else:
                misnamed += 1
            total += 1

    refused_random = 0
    max_ov_rand = []
    for _ in range(n_random):
        probe = tm.random_boundary_state(rng)
        ovs = [abs(np.vdot(p, probe)) for p in patterns]
        max_ov_rand.append(float(np.max(ovs)))
        if np.max(ovs) < tau:
            refused_random += 1

    refused_code = 0
    max_ov_code = []
    for _ in range(n_code_random):
        probe = code_random_probe(V, rng)
        ovs = [abs(np.vdot(p, probe)) for p in patterns]
        max_ov_code.append(float(np.max(ovs)))
        if np.max(ovs) < tau:
            refused_code += 1

    return {
        "s": s,
        "rho": rho,
        "sigma": sigma,
        "tau": tau,
        "noise_mode": "subspace",
        "recognition_rate": recognised / total,
        "misname_rate": misnamed / total,
        "miss_rate": missed / total,
        "refusal_rate_random": refused_random / n_random,
        "refusal_rate_code": refused_code / n_code_random,
        "mean_max_overlap_noisy": float(np.mean(max_ov_noisy)),
        "mean_max_overlap_random": float(np.mean(max_ov_rand)),
        "mean_max_overlap_code": float(np.mean(max_ov_code)),
    }


def collision_recall(V, s, tau, n_per_pair=100, seed=1234):
    """What the gate does with a superposition of two stored patterns.

    For each unordered pair (a, b), build probe = V.normalize(psi_a + psi_b)
    and record: the max overlap, which pattern the argmax names, and
    whether the gate accepts (>= tau) or refuses.  A collision is a
    genuine ambiguity -- the honest question is whether the gate admits
    it is ambiguous (refuses) or confidently names one (possibly wrong).
    """
    bulk_patterns, rho = similar_bulk_patterns(s)
    patterns = encode_patterns(V, bulk_patterns)
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in itertools.combinations(range(len(patterns)), 2):
        accepted = 0
        named_a = 0
        named_b = 0
        named_other = 0
        max_ovs = []
        for _ in range(n_per_pair):
            probe = collision_probe(V, bulk_patterns[a], bulk_patterns[b], rng)
            ovs = [abs(np.vdot(p, probe)) for p in patterns]
            m = int(np.argmax(ovs))
            max_ovs.append(float(np.max(ovs)))
            if np.max(ovs) >= tau:
                accepted += 1
                if m == a:
                    named_a += 1
                elif m == b:
                    named_b += 1
                else:
                    named_other += 1
        rows.append(
            {
                "pair": [a, b],
                "rho": rho,
                "tau": tau,
                "accept_rate": accepted / n_per_pair,
                "named_a_rate": named_a / n_per_pair,
                "named_b_rate": named_b / n_per_pair,
                "named_other_rate": named_other / n_per_pair,
                "mean_max_overlap": float(np.mean(max_ovs)),
            }
        )
    return {"s": s, "rho": rho, "tau": tau, "pairs": rows}


def best_separation(V, s, sigma, taus, seed=1234, noise_mode="boundary"):
    """Best achievable (recognition, refusal) tradeoff over a tau list.

    noise_mode "boundary": isotropic 128-dim noise (hard_recall).
    noise_mode "subspace": perturbation inside the code space
    (subspace_recall).  For each tau, returns recognition and both
    refusal rates.  Also the best (max over tau) of min(recognition,
    refusal_random) and min(recognition, refusal_code): if the best min
    is far below 1, no threshold separates the two classes.
    """
    fn = hard_recall if noise_mode == "boundary" else subspace_recall
    rows = []
    for tau in taus:
        rows.append(fn(V, s, sigma, tau, seed=seed))
    best_random = max(min(r["recognition_rate"], r["refusal_rate_random"]) for r in rows)
    best_code = max(min(r["recognition_rate"], r["refusal_rate_code"]) for r in rows)
    return {
        "s": s,
        "sigma": sigma,
        "noise_mode": noise_mode,
        "rows": rows,
        "best_min_recognition_refusal_random": float(best_random),
        "best_min_recognition_refusal_code": float(best_code),
    }
