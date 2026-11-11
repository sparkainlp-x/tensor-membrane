"""Run every measurement on the tensor membrane and print the results.

Usage: python demo.py [--seed 42] [--write]
  --write writes results.json next to this file.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_membrane import model as tm

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    V, U, gates = tm.encoding(seed=args.seed)
    print(f"tensor-membrane v{tm.__version__} — seed {args.seed}")
    print(f"network: {len(gates)} tensors (layers 6/7/6), "
          f"bulk {tm.N_BULK} qubits -> boundary {tm.N_BOUNDARY} qubits")
    print(f"encoding V: {V.shape[0]}x{V.shape[1]}")

    # 1. Isometry
    dev, svals = tm.check_isometry(V)
    print("\n--- isometry (bulk <-> boundary) ---")
    print(f"max |V^dagger V - I| = {dev:.3e}")
    print(f"singular values of V: min {svals.min():.6f}, max {svals.max():.6f}")
    # boundary states orthogonal?
    G = V.conj().T @ V
    off = np.max(np.abs(G - np.eye(tm.BULK_DIM)))
    print(f"max |<psi_b|psi_b'>| for b != b' = {off:.3e}")

    # 2. Delocalisation
    deloc = tm.delocalisation(V)
    print("\n--- delocalisation (one boundary qubit at a time) ---")
    print("qubit | mean trace dist. between bulk states | mean entropy (bits)")
    for p in deloc["per_qubit"]:
        print(f"  {p['qubit']}   | {p['mean_trace_distance']:.4f} | {p['mean_entropy_bits']:.4f}")
    print(f"overall mean trace distance: {deloc['overall_mean_trace_distance']:.4f} "
          f"(0 = no single qubit carries the bulk)")
    print(f"overall mean single-qubit entropy: {deloc['overall_mean_entropy_bits']:.4f} bits "
          f"(1 = maximally mixed)")

    # 3. Erasure recovery
    eras = tm.erasure_recovery(V, max_k=5)
    print("\n--- erasure recovery (trace distance between bulk states on survivors) ---")
    print("erased k | subsets | worst-case mean dist. | mean over subsets")
    for e in eras:
        print(f"  {e['erased_k']}      | {e['n_subsets']:>3}     | {e['min_mean_trace_distance']:.4f} | "
              f"{e['mean_mean_trace_distance']:.4f}")
    print("1 = perfectly distinguishable (recoverable); 0 = indistinguishable.")

    # 4. Overlap recall
    print("\n--- overlap recall (stored = 4 bulk basis states) ---")
    print("sigma | tau | recognition | refusal | mean max overlap noisy | random")
    recall_rows = []
    for sigma in [0.0, 0.05, 0.10, 0.15, 0.30, 0.50]:
        for tau in [0.5]:
            r = tm.overlap_recall(V, sigma, tau)
            recall_rows.append(r)
            print(f" {sigma:.2f}  | {tau:.1f} | {r['recognition_rate']:.3f} | "
                  f"{r['refusal_rate']:.3f} | {r['mean_max_overlap_noisy']:.3f} | "
                  f"{r['mean_max_overlap_random']:.3f}")
    # tau sweep at one noise level
    print("\ntau sweep at sigma = 0.15:")
    print("tau | recognition | refusal")
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r = tm.overlap_recall(V, 0.15, tau)
        recall_rows.append(r)
        print(f" {tau:.1f} | {r['recognition_rate']:.3f} | {r['refusal_rate']:.3f}")

    if args.write:
        results = {
            "version": tm.__version__,
            "seed": args.seed,
            "layers": [len(p) for p in tm.LAYER_PAIRS],
            "bulk_qubits": tm.N_BULK,
            "boundary_qubits": tm.N_BOUNDARY,
            "isometry_max_deviation": dev,
            "singular_values": [float(s) for s in svals],
            "boundary_orthogonality_max_offdiag": float(off),
            "delocalisation": deloc,
            "erasure_recovery": eras,
            "overlap_recall": recall_rows,
        }
        path = os.path.join(os.path.dirname(HERE), "results", "results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
