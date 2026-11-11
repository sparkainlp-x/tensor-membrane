"""Tests for the hard overlap recall.

Pins the design (pattern construction, encoding preserves overlap,
determinism) and the measured behaviour: isotropic boundary noise does
NOT break recognition, subspace noise breaks it by misnaming, no tau
separates real from code-random at high similarity + subspace noise,
and the gate has no collision detection.
"""

import numpy as np

from tensor_membrane import hard_recall as hr
from tensor_membrane import model as tm

V, _, _ = tm.encoding(seed=42)


def test_pattern_construction_endpoints():
    pats, rho = hr.similar_bulk_patterns(0.0)
    assert abs(rho) < 1e-12, rho
    for i in range(4):
        assert abs(np.linalg.norm(pats[i]) - 1.0) < 1e-12
        for j in range(i + 1, 4):
            assert abs(np.vdot(pats[i], pats[j])) < 1e-12
    pats, rho = hr.similar_bulk_patterns(1.0)
    assert abs(rho - 1.0) < 1e-12, rho


def test_pattern_overlap_monotone_in_s():
    rhos = [hr.similar_bulk_patterns(s)[1] for s in
            [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]]
    for a, b in zip(rhos, rhos[1:]):
        assert b > a, (a, b)
    assert rhos[-1] > 0.9


def test_pattern_construction_rejects_bad_s():
    for bad in [-0.1, 1.1, 2.0]:
        try:
            hr.similar_bulk_patterns(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"s={bad} did not raise")


def test_encoding_preserves_overlap():
    for s in [0.0, 0.5, 0.9]:
        bulk, rho_bulk = hr.similar_bulk_patterns(s)
        pats = hr.encode_patterns(V, bulk)
        assert abs(np.linalg.norm(pats[0]) - 1.0) < 1e-12
        ovs = [abs(np.vdot(pats[i], pats[j])) for i in range(4) for j in range(i + 1, 4)]
        assert abs(float(np.mean(ovs)) - rho_bulk) < 1e-12


def test_hard_recall_deterministic():
    r1 = hr.hard_recall(V, 0.7, 0.15, 0.5, seed=99)
    r2 = hr.hard_recall(V, 0.7, 0.15, 0.5, seed=99)
    assert r1 == r2


def test_isotropic_noise_never_breaks_recognition():
    # The similarity knob does NOT break recognition under isotropic
    # 128-dim boundary noise: the noise is diluted across 128 dims.
    # Even at rho=0.925, sigma=0.5, recognition stays 1.000.
    r = hr.hard_recall(V, 0.95, 0.50, 0.5)
    assert r["recognition_rate"] == 1.0
    assert r["misname_rate"] == 0.0
    assert r["refusal_rate_random"] == 1.0


def test_code_random_accepted_at_natural_tau():
    # At tau=0.5 the gate accepts (nearly) every code-random probe,
    # even at rho=0: the gate refuses isotropic random, not unfamiliar
    # structure in the code subspace.
    r = hr.hard_recall(V, 0.0, 0.15, 0.5)
    assert r["refusal_rate_code"] < 0.05, r["refusal_rate_code"]
    assert r["mean_max_overlap_code"] > 0.6


def test_subspace_noise_breaks_recognition_by_misnaming():
    # At (rho=0.864, sigma_sub=0.5) recognition drops to ~0.80 and the
    # failures are misnames, not misses: the gate confidently names the
    # wrong pattern instead of refusing.
    r = hr.subspace_recall(V, 0.9, 0.5, 0.5)
    assert r["recognition_rate"] < 0.9, r["recognition_rate"]
    assert r["misname_rate"] > 0.1, r["misname_rate"]
    assert r["miss_rate"] < 0.05, r["miss_rate"]
    # Orthogonal patterns hold longer: at rho=0, sigma_sub=0.5 it is
    # still perfect.
    r0 = hr.subspace_recall(V, 0.0, 0.5, 0.5)
    assert r0["recognition_rate"] == 1.0


def test_subspace_breaking_point_moves_with_similarity():
    # Higher similarity breaks at lower subspace noise.
    r_hi = hr.subspace_recall(V, 0.95, 0.3, 0.5)
    r_lo = hr.subspace_recall(V, 0.0, 0.3, 0.5)
    assert r_hi["recognition_rate"] < 1.0, r_hi["recognition_rate"]
    assert r_lo["recognition_rate"] == 1.0


def test_no_tau_separates_at_high_similarity_subspace_noise():
    # At (rho=0.864, sigma_sub=0.5) no tau gets both recognition and
    # code-refusal above 0.85: the gate fails to separate the classes.
    # (Measured best min is 0.805.)
    sep = hr.best_separation(V, 0.9, 0.5, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             noise_mode="subspace")
    assert sep["best_min_recognition_refusal_code"] < 0.85
    # But at low similarity + low noise it does separate.
    sep0 = hr.best_separation(V, 0.0, 0.15, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                              noise_mode="subspace")
    assert sep0["best_min_recognition_refusal_code"] > 0.9


def test_isotropic_separation_breaks_between_sigma_07_and_10():
    sep07 = hr.best_separation(V, 0.9, 0.7, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    sep10 = hr.best_separation(V, 0.9, 1.0, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert sep07["best_min_recognition_refusal_code"] > 0.85
    assert sep10["best_min_recognition_refusal_code"] < 0.7


def test_collision_always_accepted_and_tied():
    # A superposition of two stored patterns is always accepted and
    # named as one of the pair -- never refused, never "other".
    for s in [0.0, 0.5, 0.9]:
        c = hr.collision_recall(V, s, 0.5)
        assert c["rho"] == hr.similar_bulk_patterns(s)[1]
        for row in c["pairs"]:
            assert row["accept_rate"] == 1.0
            assert row["named_other_rate"] == 0.0
            assert row["named_a_rate"] + row["named_b_rate"] == 1.0
            # mean max overlap matches (1+rho)/sqrt(2+2rho)
            expect = (1 + c["rho"]) / np.sqrt(2 + 2 * c["rho"])
            assert abs(row["mean_max_overlap"] - expect) < 1e-9
    # The tie itself: the two colliding patterns have exactly equal
    # overlap (to floating point), so the naming is arbitrary.
    bulk, rho = hr.similar_bulk_patterns(0.5)
    pats = hr.encode_patterns(V, bulk)
    probe = hr.collision_probe(V, bulk[0], bulk[1], np.random.default_rng(0))
    ov0 = abs(np.vdot(pats[0], probe))
    ov1 = abs(np.vdot(pats[1], probe))
    assert abs(ov0 - ov1) < 1e-12


def test_subspace_recognition_degrades_with_similarity():
    # At fixed subspace noise, higher similarity means worse recognition.
    recs = [
        hr.subspace_recall(V, s, 0.7, 0.5)["recognition_rate"]
        for s in [0.0, 0.5, 0.9]
    ]
    assert recs[0] >= recs[1] >= recs[2], recs
    assert recs[0] == 1.0
    assert recs[2] < 0.7


def test_robustness_other_network_seed():
    Vn, _, _ = tm.encoding(seed=7)
    r = hr.subspace_recall(Vn, 0.9, 0.5, 0.5)
    assert r["recognition_rate"] < 0.9, r["recognition_rate"]
    assert r["misname_rate"] > 0.1
    assert r["refusal_rate_random"] == 1.0
