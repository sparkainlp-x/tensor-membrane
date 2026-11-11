# Changelog

## 0.1.0 — 2026-11-11

First release. A holographic tensor network as a working membrane
(synthetic, stdlib + NumPy, seeded, reproducible, AGPL-3.0-only).

- **Base exploration** (`tensor_membrane/model.py`): 7 qubits — 2 bulk
  logical, 5 ancillas in |0⟩ — encoded by 19 random 2-qubit unitaries in
  3 layers of 6/7/6, matching the diagram. Measured (seed 42): the
  encoding is an isometry (max |V†V − I| = 1.7e-15); no single boundary
  qubit carries the bulk (mean trace distance 0.146, mean entropy 0.955
  bits); erasure threshold 2 (worst case: k=1 → 0.973, k=2 → 0.919,
  k=3 → 0.737, k=4 → 0.416, k=5 → 0.213); overlap recall 1.000/1.000 at
  every noise level and threshold tested.
- **Real erasure decoder** (`tensor_membrane/decoder.py`): least-squares
  channel inversion (pseudoinverse + projection onto the physical state
  space). Measured (seed 42): with exact input the bulk is recoverable
  for every k ≤ 5; with noisy input the threshold is noise-dependent —
  k=4 at ε=0.01, k=3 at ε=0.1, k=1 at ε=0.5. The mechanism is the
  channel conditioning (worst-case cond 1.3 at k=1, 1697 at k=5).
- **Hard overlap recall** (`tensor_membrane/hard_recall.py`):
  non-orthogonal stored patterns (similarity s, pairwise overlap ρ),
  code-random probes, subspace noise, collision probes. Measured
  (seed 42): similarity does not break recognition under boundary noise
  (1.000 at every ρ, σ); subspace noise breaks it by confident
  misnaming (at ρ=0.864, σ=0.5: recognition 0.805, misname 0.195, miss
  0.000); no threshold separates real from code-random at high
  similarity (best min 0.805 at ρ=0.864, σ=0.5; 0.335 at ρ=0.925,
  σ=1.5); the gate has no collision detection (a superposition of two
  stored patterns is always accepted, 1.000 at every ρ).
- 38 tests, all passing. Results in `results/` (seed 42).
