"""Run the real decoder measurement and print the results.

Usage: python demo_decoder.py [--seed 42] [--write]
  --write writes decoder_results.json next to this file.

Measures, for each erasure count k = 0..5 (worst case over all subsets,
and mean):
  - channel conditioning (rank, worst-case condition number);
  - recovery fidelity with EXACT reduced-state input (structural);
  - recovery fidelity with NOISY reduced-state input at several relative
    noise levels epsilon (practical; sigma = epsilon / d_S).
Compares against the distinguishability numbers from the earlier
exploration (trace distance on survivors): k=1 -> 0.973, k=2 -> 0.919,
k=3 -> 0.737, k=4 -> 0.416, k=5 -> 0.213 (worst case).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_membrane import model as tm
from tensor_membrane import decoder as dec

HERE = os.path.dirname(os.path.abspath(__file__))

DISTINGUISHABILITY_WORST = {0: 1.000, 1: 0.973, 2: 0.919, 3: 0.737, 4: 0.416, 5: 0.213}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    V, _, _ = tm.encoding(seed=args.seed)
    print(f"tensor-membrane decoder v{tm.__version__} — seed {args.seed}")
    print("decoder: least-squares channel inversion (pseudoinverse) + "
          "projection onto the physical state space")
    print("test states: 4 bulk basis states + 16 Haar-random pure bulk states")

    res = dec.measure_recovery(V, max_k=5, n_random=16,
                               epsilons=(0.0, 0.01, 0.1, 0.5))
    eps = res["epsilons"]

    print("\n--- channel conditioning (worst case over subsets) ---")
    print("k | subsets | min rank | max cond")
    for row in res["per_k"]:
        print(f"{row['erased_k']} | {row['n_subsets']:>7} | {row['min_rank']:>8} | "
              f"{row['max_cond']:.4g}")

    print("\n--- recovery fidelity, EXACT input (epsilon = 0) ---")
    print("k | worst | mean  | distinguishability (trace dist., worst)")
    for row in res["per_k"]:
        f = row["by_epsilon"]["0.0"]
        print(f"{row['erased_k']} | {f['worst_fidelity']:.4f} | {f['mean_fidelity']:.4f} | "
              f"{DISTINGUISHABILITY_WORST[row['erased_k']]:.3f}")

    for e in [0.01, 0.1, 0.5]:
        print(f"\n--- recovery fidelity, NOISY input (epsilon = {e}, "
              f"sigma = epsilon / d_S) ---")
        print("k | worst | mean")
        for row in res["per_k"]:
            f = row["by_epsilon"][str(e)]
            print(f"{row['erased_k']} | {f['worst_fidelity']:.4f} | {f['mean_fidelity']:.4f}")

    # Headline: threshold comparison. Recoverable = worst-case fidelity >= 0.90.
    print("\n--- threshold (worst-case fidelity >= 0.90) ---")
    for e in eps:
        ok = [row["erased_k"] for row in res["per_k"]
              if row["by_epsilon"][str(e)]["worst_fidelity"] >= 0.90]
        thr = max(ok) if ok else None
        print(f"epsilon = {e}: threshold k = {thr} "
              f"(distinguishability threshold was k = 2)")

    if args.write:
        payload = {
            "version": tm.__version__,
            "seed": args.seed,
            "method": "least-squares channel inversion (pseudoinverse, rcond=1e-12) "
                      "+ projection onto the physical state space",
            "noise_model": "Hermitian Gaussian added to the reduced state, "
                           "sigma = epsilon / d_S",
            "n_random_states": 16,
            "distinguishability_worst": DISTINGUISHABILITY_WORST,
            "per_k": res["per_k"],
        }
        path = os.path.join(os.path.dirname(HERE), "results", "decoder_results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
