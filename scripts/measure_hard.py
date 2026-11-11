"""Sweep the hard recall: similarity x noise at tau=0.5, tau sweeps at key
points, and a robustness repeat on other network seeds. Writes
hard_results.json next to this file.

Usage: python measure_hard.py [--seed 42] [--write]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_membrane import hard_recall as hr
from tensor_membrane import model as tm

HERE = os.path.dirname(os.path.abspath(__file__))

S_GRID = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
SIGMA_GRID = [0.0, 0.15, 0.30, 0.50]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    V, _, _ = tm.encoding(seed=args.seed)
    print(f"hard recall — network seed {args.seed}")
    print("s | rho | sigma | tau | recog | misname | miss | refuse-rand | refuse-code | "
          "mean max ov noisy | rand | code")

    grid = []
    for s in S_GRID:
        for sigma in SIGMA_GRID:
            r = hr.hard_recall(V, s, sigma, 0.5)
            grid.append(r)
            print(f"{s:.2f} | {r['rho']:.3f} | {sigma:.2f} | 0.5 | {r['recognition_rate']:.3f} | "
                  f"{r['misname_rate']:.3f} | {r['miss_rate']:.3f} | {r['refusal_rate_random']:.3f} | "
                  f"{r['refusal_rate_code']:.3f} | {r['mean_max_overlap_noisy']:.3f} | "
                  f"{r['mean_max_overlap_random']:.3f} | {r['mean_max_overlap_code']:.3f}")

    # Tau sweep / best separation at key points.
    print("\n--- best separation over tau (taus 0.3..0.9) ---")
    print("s | rho | sigma | best min(recog, refuse-rand) | best min(recog, refuse-code)")
    separations = []
    for (s, sigma) in [(0.0, 0.15), (0.5, 0.15), (0.7, 0.15), (0.85, 0.15),
                       (0.9, 0.15), (0.9, 0.30), (0.95, 0.15)]:
        sep = hr.best_separation(V, s, sigma, TAUS)
        separations.append(
            {
                "s": sep["s"],
                "sigma": sep["sigma"],
                "best_min_recognition_refusal_random": sep["best_min_recognition_refusal_random"],
                "best_min_recognition_refusal_code": sep["best_min_recognition_refusal_code"],
                "rows": sep["rows"],
            }
        )
        rho = sep["rows"][0]["rho"]
        print(f"{s:.2f} | {rho:.3f} | {sigma:.2f} | "
              f"{sep['best_min_recognition_refusal_random']:.3f} | "
              f"{sep['best_min_recognition_refusal_code']:.3f}")

    # Robustness: repeat a few grid cells on other network seeds.
    print("\n--- robustness (other network seeds) ---")
    robust = []
    for net_seed in [7, 123]:
        Vn, _, _ = tm.encoding(seed=net_seed)
        for (s, sigma) in [(0.7, 0.15), (0.9, 0.15), (0.9, 0.30)]:
            r = hr.hard_recall(Vn, s, sigma, 0.5)
            robust.append({"network_seed": net_seed, **r})
            print(f"net {net_seed} | s {s:.2f} | rho {r['rho']:.3f} | sigma {sigma:.2f} | "
                  f"recog {r['recognition_rate']:.3f} | misname {r['misname_rate']:.3f} | "
                  f"refuse-rand {r['refusal_rate_random']:.3f} | "
                  f"refuse-code {r['refusal_rate_code']:.3f}")

    # Subspace noise: perturb the bulk state before encoding. This is the
    # noise that can actually confuse similar stored patterns.
    print("\n--- subspace noise (perturb inside the code space) ---")
    print("s | rho | sigma_sub | tau | recog | misname | miss | refuse-rand | refuse-code | "
          "mean max ov noisy")
    sub_grid = []
    for s in S_GRID:
        for sigma_sub in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]:
            r = hr.subspace_recall(V, s, sigma_sub, 0.5)
            sub_grid.append(r)
            print(f"{s:.2f} | {r['rho']:.3f} | {sigma_sub:.2f} | 0.5 | {r['recognition_rate']:.3f} | "
                  f"{r['misname_rate']:.3f} | {r['miss_rate']:.3f} | {r['refusal_rate_random']:.3f} | "
                  f"{r['refusal_rate_code']:.3f} | {r['mean_max_overlap_noisy']:.3f}")

    print("\n--- subspace best separation over tau (taus 0.3..0.9) ---")
    sub_sep = []
    for (s, sigma_sub) in [(0.0, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5),
                           (0.9, 1.0), (0.95, 1.0), (0.95, 1.5)]:
        rows = [hr.subspace_recall(V, s, sigma_sub, tau) for tau in TAUS]
        best_random = max(min(r["recognition_rate"], r["refusal_rate_random"]) for r in rows)
        best_code = max(min(r["recognition_rate"], r["refusal_rate_code"]) for r in rows)
        sub_sep.append(
            {
                "s": s,
                "sigma": sigma_sub,
                "best_min_recognition_refusal_random": float(best_random),
                "best_min_recognition_refusal_code": float(best_code),
                "rows": rows,
            }
        )
        rho = rows[0]["rho"]
        print(f"{s:.2f} | {rho:.3f} | {sigma_sub:.2f} | {best_random:.3f} | {best_code:.3f}")

    # Extended isotropic noise: where does the boundary-noise separation break?
    print("\n--- isotropic noise, extended sigma (best separation over tau) ---")
    ext_sep = []
    for sigma in [0.5, 0.7, 1.0, 1.5]:
        sep = hr.best_separation(V, 0.9, sigma, TAUS)
        ext_sep.append(
            {
                "s": sep["s"],
                "sigma": sep["sigma"],
                "best_min_recognition_refusal_random": sep["best_min_recognition_refusal_random"],
                "best_min_recognition_refusal_code": sep["best_min_recognition_refusal_code"],
                "rows": sep["rows"],
            }
        )
        print(f"s 0.90 | sigma {sigma:.2f} | best min(rand) "
              f"{sep['best_min_recognition_refusal_random']:.3f} | best min(code) "
              f"{sep['best_min_recognition_refusal_code']:.3f}")

    # Collision probes: superpositions of two stored patterns.
    print("\n--- collision probes (superposition of two stored patterns) ---")
    print("s | rho | tau | pair | accept | named_a | named_b | named_other | mean max ov")
    collisions = []
    for s in [0.0, 0.5, 0.9]:
        c = hr.collision_recall(V, s, 0.5)
        collisions.append(c)
        for row in c["pairs"]:
            print(f"{s:.2f} | {row['rho']:.3f} | 0.5 | {row['pair']} | {row['accept_rate']:.3f} | "
                  f"{row['named_a_rate']:.3f} | {row['named_b_rate']:.3f} | "
                  f"{row['named_other_rate']:.3f} | {row['mean_max_overlap']:.3f}")

    if args.write:
        results = {
            "version": tm.__version__,
            "network_seed": args.seed,
            "s_grid": S_GRID,
            "sigma_grid": SIGMA_GRID,
            "taus": TAUS,
            "grid": grid,
            "separations": separations,
            "robustness": robust,
            "subspace_grid": sub_grid,
            "subspace_separations": sub_sep,
            "extended_isotropic_separations": ext_sep,
            "collisions": collisions,
        }
        path = os.path.join(os.path.dirname(HERE), "results", "hard_results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
