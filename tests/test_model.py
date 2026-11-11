"""Tests for the tensor membrane. Run: python -m pytest tests.py -q
(or: python tests.py — it uses unittest and runs standalone too).
"""

import unittest

import numpy as np

from tensor_membrane import model as tm


class TestNetwork(unittest.TestCase):
    def test_gate_embedding_is_unitary(self):
        rng = np.random.default_rng(0)
        for (q1, q2) in [(0, 1), (0, 6), (3, 5)]:
            g = tm.random_unitary_2q(rng)
            G = tm.embed_gate(g, q1, q2)
            dev = np.max(np.abs(G.conj().T @ G - np.eye(128)))
            self.assertLess(dev, 1e-12)

    def test_deterministic_same_seed(self):
        V1, _, _ = tm.encoding(seed=42)
        V2, _, _ = tm.encoding(seed=42)
        self.assertTrue(np.allclose(V1, V2))
        V3, _, _ = tm.encoding(seed=7)
        self.assertFalse(np.allclose(V1, V3))

    def test_layer_counts(self):
        self.assertEqual([len(p) for p in tm.LAYER_PAIRS], [6, 7, 6])
        _, _, gates = tm.encoding(seed=42)
        self.assertEqual(len(gates), 19)


class TestIsometry(unittest.TestCase):
    def test_encoding_is_isometry(self):
        V, _, _ = tm.encoding(seed=42)
        dev, svals = tm.check_isometry(V)
        self.assertLess(dev, 1e-12)
        self.assertTrue(np.allclose(svals, 1.0, atol=1e-12))

    def test_boundary_states_orthogonal(self):
        V, _, _ = tm.encoding(seed=42)
        G = V.conj().T @ V
        off = np.max(np.abs(G - np.eye(tm.BULK_DIM)))
        self.assertLess(off, 1e-12)


class TestDelocalisation(unittest.TestCase):
    def test_no_single_qubit_carries_the_bulk(self):
        V, _, _ = tm.encoding(seed=42)
        d = tm.delocalisation(V)
        # measured overall mean trace distance 0.146 (seed 42); require < 0.25
        self.assertLess(d["overall_mean_trace_distance"], 0.25)
        # every single qubit is near-maximally mixed (measured 0.92..0.99)
        for p in d["per_qubit"]:
            self.assertGreater(p["mean_entropy_bits"], 0.9)
            self.assertLess(p["mean_trace_distance"], 0.30)

    def test_single_qubit_entropy_near_max(self):
        V, _, _ = tm.encoding(seed=42)
        d = tm.delocalisation(V)
        # measured overall mean 0.955 bits; require > 0.9
        self.assertGreater(d["overall_mean_entropy_bits"], 0.9)


class TestErasure(unittest.TestCase):
    def test_one_erasure_near_perfect(self):
        V, _, _ = tm.encoding(seed=42)
        eras = tm.erasure_recovery(V, max_k=2)
        # measured worst-case k=1: 0.973; require > 0.95
        self.assertGreater(eras[1]["min_mean_trace_distance"], 0.95)

    def test_two_erasures_hold(self):
        V, _, _ = tm.encoding(seed=42)
        eras = tm.erasure_recovery(V, max_k=2)
        # measured worst-case k=2: 0.919; require > 0.90
        self.assertGreater(eras[2]["min_mean_trace_distance"], 0.90)

    def test_three_erasures_degrade(self):
        V, _, _ = tm.encoding(seed=42)
        eras = tm.erasure_recovery(V, max_k=3)
        # measured worst-case k=3: 0.737; require < 0.80 (degraded, not recovered)
        self.assertLess(eras[3]["min_mean_trace_distance"], 0.80)

    def test_monotone_decay(self):
        V, _, _ = tm.encoding(seed=42)
        eras = tm.erasure_recovery(V, max_k=5)
        worst = [e["min_mean_trace_distance"] for e in eras]
        for a, b in zip(worst, worst[1:]):
            self.assertGreaterEqual(a, b)


class TestOverlapRecall(unittest.TestCase):
    def test_clean_recognition_and_refusal(self):
        V, _, _ = tm.encoding(seed=42)
        r = tm.overlap_recall(V, sigma=0.0, tau=0.5)
        self.assertEqual(r["recognition_rate"], 1.0)
        self.assertEqual(r["refusal_rate"], 1.0)

    def test_noisy_recognition_and_refusal(self):
        V, _, _ = tm.encoding(seed=42)
        r = tm.overlap_recall(V, sigma=0.15, tau=0.5)
        self.assertEqual(r["recognition_rate"], 1.0)
        self.assertEqual(r["refusal_rate"], 1.0)
        # random probes overlap far below noisy probes
        self.assertLess(r["mean_max_overlap_random"], 0.2)
        self.assertGreater(r["mean_max_overlap_noisy"], 0.9)

    def test_large_noise_still_recognised(self):
        V, _, _ = tm.encoding(seed=42)
        r = tm.overlap_recall(V, sigma=0.5, tau=0.5)
        self.assertEqual(r["recognition_rate"], 1.0)
        self.assertEqual(r["refusal_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
