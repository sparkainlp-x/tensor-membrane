# Tensor membrane

Jean's holographic tensor network, built as a working membrane model.
Local exploration only — not a real quantum device, not a real code.

## The diagram

```
Boundary physical qubits
o   o   o   o   o   o   o
 \ / \ / \ / \ / \ / \ /
  T---T---T---T---T---T      tensor layer
 / \ / \ / \ / \ / \ / \
T---T---T---T---T---T---T    tensor layer
 \ / \ / \ / \ / \ / \ /
  T---T---T---T---T---T      tensor layer
   \  |  /
bulk logical qubits
```

- **Boundary physical qubits** (top): 7 qubits.
- **Tensor layers** (middle): three layers — 6, then 7, then 6 tensors.
- **Bulk logical qubits** (bottom): the protected information, reached by fan-in.

The bulk is the message, the boundary is the hologram, the tensors are
the lens. A bulk logical qubit is delocalised across the boundary —
that is the error correction. Same "holographic" as the membrane work
(information stored in a pattern, recoverable by overlap), in the
quantum error-correction language.

## The model

`tensor_membrane/model.py` (stdlib + NumPy only, seeded, reproducible).

- 7 qubits total. Qubits 0 and 1 are the bulk (2 logical qubits, dim 4).
  Qubits 2..6 are ancillas, initialised to |0⟩.
- The network is 19 random 2-qubit unitaries (the "tensors"), in 3 layers
  of 6, 7, 6 — matching the diagram's layer counts. Each unitary is drawn
  from a seeded RNG (QR of a complex Gaussian). The pairing pattern is our
  concrete choice (the sketch does not specify exact pairs): layer 1 fans
  out from the bulk, layer 2 mixes across, layer 3 mixes near the boundary.
- The 19 unitaries compose to a 7-qubit unitary U (128x128). The encoding
  V is U restricted to the ancillas-in-|0⟩ subspace (columns for
  |b0 b1 0 0 0 0 0⟩). V is therefore an isometry by construction.

## Measured results — part 1: the base exploration (seed 42)

Run: `python scripts/demo.py` — reproduces `results/results.json`.

### Isometry (bulk ↔ boundary)

| Quantity | Measured |
|---|---|
| max \|V†V − I\| | 1.7e-15 (machine precision) |
| Singular values of V | all 1.000000 |
| max \|⟨ψ_b\|ψ_b′⟩\|, b ≠ b′ | 1.7e-15 (boundary states orthogonal) |

The encoding preserves inner products exactly. The bulk maps to an
orthonormal set of boundary states.

### Delocalisation (one boundary qubit at a time)

| Boundary qubit | mean trace distance between bulk states | mean entropy |
|---|---|---|
| 0 | 0.106 | 0.978 bits |
| 1 | 0.124 | 0.948 bits |
| 2 | 0.163 | 0.952 bits |
| 3 | 0.223 | 0.923 bits |
| 4 | 0.146 | 0.969 bits |
| 5 | 0.089 | 0.986 bits |
| 6 | 0.171 | 0.932 bits |
| **overall mean** | **0.146** | **0.955 bits** |

Trace distance near 0 means that qubit alone cannot tell the bulk states
apart; entropy near 1 bit means it is nearly maximally mixed. No single
boundary qubit carries the bulk — the information lives in the pattern
across all 7, not in any one of them. That is the delocalisation the
diagram promises, measured.

### Erasure recovery — distinguishability (worst case over all subsets)

| Erased k | subsets | worst-case mean trace distance | mean over subsets |
|---|---|---|---|
| 0 | 1 | 1.000 | 1.000 |
| 1 | 7 | 0.973 | 0.985 |
| 2 | 21 | 0.919 | 0.936 |
| 3 | 35 | 0.737 | 0.766 |
| 4 | 35 | 0.416 | 0.468 |
| 5 | 21 | 0.213 | 0.269 |

Trace distance near 1 = the bulk states stay distinguishable from the
survivors (recoverable); near 0 = indistinguishable (lost).

**Threshold: 2 erasures.** One erasure is essentially free (worst 0.97),
two hold up (worst 0.92), three degrade (worst 0.74), four and five fail.
The threshold uses "recoverable" = worst-case mean trace distance ≥ 0.90.
It matches the quantum Singleton bound for a [[7,2]] code ((7−2)/2 = 2.5,
so 2 erasures) — the random network achieves what the bound allows.

### Overlap recall (membrane-brain style; stored = 4 bulk basis states)

| σ (noise) | τ | recognition | refusal | mean max overlap: noisy | random |
|---|---|---|---|---|---|
| 0.00 | 0.5 | 1.000 | 1.000 | 1.000 | 0.121 |
| 0.05 | 0.5 | 1.000 | 1.000 | 0.999 | 0.121 |
| 0.10 | 0.5 | 1.000 | 1.000 | 0.995 | 0.121 |
| 0.15 | 0.5 | 1.000 | 1.000 | 0.989 | 0.121 |
| 0.30 | 0.5 | 1.000 | 1.000 | 0.958 | 0.121 |
| 0.50 | 0.5 | 1.000 | 1.000 | 0.895 | 0.121 |

Tau sweep at σ = 0.15: recognition 1.000 and refusal 1.000 for every
τ in [0.3, 0.8]. The separation is so large (noisy ~0.99 vs random
~0.12) that the threshold choice does not matter. Recognition stays
perfect even at σ = 0.50, because the noise is spread across 128
dimensions and barely overlaps any stored pattern.

### Robustness across seeds (part 1)

Same measurements on seeds 7, 123, 2026:

| Seed | deloc. mean dist. | entropy | erasure worst k=1 / 2 / 3 |
|---|---|---|---|
| 7 | 0.107 | 0.973 | 0.981 / 0.924 / 0.719 |
| 123 | 0.125 | 0.954 | 0.973 / 0.901 / 0.712 |
| 2026 | 0.144 | 0.958 | 0.977 / 0.919 / 0.732 |
| 42 | 0.146 | 0.955 | 0.973 / 0.919 / 0.737 |

The threshold (2 erasures) and the delocalisation hold on every seed.
Recall is 1.000 / 1.000 on every seed.

## Measured results — part 2: a real erasure decoder (seed 42)

Run: `python scripts/demo_decoder.py` — reproduces
`results/decoder_results.json`.

Part 1 measured **distinguishability**: the trace distance between bulk
states on the surviving qubits. That is a necessary condition for
recovery, not a demonstrated decoding procedure. `tensor_membrane/decoder.py`
builds an actual decoder and measures **recovery fidelity**: the overlap
between the recovered bulk state and the true bulk state.

**Method — least-squares channel inversion.** The erasure is a linear map
(a quantum channel) on the bulk: N(ρ) = Tr_E[VρV†], where V is the
encoding isometry and E is the erased set. The map is written as a matrix
A acting on the vectorised 4x4 input; A is built column by column from the
reduced operators of |ψ_i⟩⟨ψ_j|. The decoder inverts A with a
pseudoinverse (SVD, relative cutoff 1e-12) and projects the output onto
the physical state space (symmetrise, clip negative eigenvalues,
renormalise trace to 1). The decoder knows V and which qubits were erased
(standard for an erasure channel). Test states: the 4 bulk basis states
plus 16 Haar-random pure bulk states. Two input regimes: **exact** (the
decoder receives the exact reduced state — structural recoverability) and
**noisy** (the reduced state plus Hermitian Gaussian noise, σ = ε/d_S, a
proxy for finite-sample tomography error — practical recoverability).

### Channel conditioning (worst case over subsets)

| k | subsets | min rank | max cond |
|---|---------|----------|----------|
| 0 | 1 | 16 | 1 |
| 1 | 7 | 16 | 1.26 |
| 2 | 21 | 16 | 1.70 |
| 3 | 35 | 16 | 2.96 |
| 4 | 35 | 16 | 7.45 |
| 5 | 21 | 16 | 1697 |

The map is full rank (16) for every k ≤ 5 on every subset. But the
condition number explodes at k=5: the worst subset (0,2,3,4,5) has
smallest singular value 0.000619 (largest 1.05). That tiny direction is
what makes k=5 fragile — noise is amplified ~1600× along it. A typical
k=5 subset has cond ~102.

### Recovery fidelity

| k | exact worst | exact mean | ε=0.01 worst | ε=0.01 mean | ε=0.1 worst | ε=0.1 mean | ε=0.5 worst | ε=0.5 mean |
|---|-------------|------------|--------------|-------------|-------------|------------|-------------|------------|
| 0 | 1.0000 | 1.0000 | 0.9997 | 0.9998 | 0.9968 | 0.9984 | 0.9862 | 0.9911 |
| 1 | 1.0000 | 1.0000 | 0.9988 | 0.9995 | 0.9893 | 0.9950 | 0.9458 | 0.9767 |
| 2 | 1.0000 | 1.0000 | 0.9966 | 0.9986 | 0.9626 | 0.9862 | 0.8621 | 0.9364 |
| 3 | 1.0000 | 1.0000 | 0.9901 | 0.9962 | 0.9161 | 0.9641 | 0.6391 | 0.8341 |
| 4 | 1.0000 | 1.0000 | 0.9703 | 0.9882 | 0.7839 | 0.8905 | 0.1345 | 0.6015 |
| 5 | 1.0000 | 1.0000 | 0.0661 | 0.7980 | 0.0147 | 0.4137 | 0.0152 | 0.2771 |

### Threshold (worst-case fidelity ≥ 0.90)

| input | threshold k | vs distinguishability (k = 2) |
|-------|-------------|-------------------------------|
| exact (ε = 0) | 5 | BETTER — the map is injective everywhere |
| ε = 0.01 | 4 | BETTER |
| ε = 0.1 | 3 | BETTER |
| ε = 0.5 | 1 | WORSE |

**The distinguishability threshold (k = 2) was pessimistic.** Trace
distance is a pairwise-discrimination measure; the decoder uses the FULL
linear structure of the channel, so it recovers more than pairwise
discrimination suggests. With exact knowledge of the reduced state, the
bulk is recoverable for every k ≤ 5.

**But the practical threshold is not a single number — it depends on how
well you can estimate the reduced state.** At k ≤ 4 the channel is well
conditioned (cond ≤ 7.5), so moderate estimation noise barely hurts. At
k = 5 the worst subset has cond 1697, so even small noise destroys the
recovery on that subset (worst fidelity 0.066 at ε = 0.01, while the mean
is 0.798 — the failure is subset-dependent). At high noise (ε = 0.5) even
k = 2 fails worst-case (0.862).

So: **structurally the threshold is 5, practically it is 4 at clean
estimation, 3 at moderate noise, 1 at noisy estimation.** The honest
single number, if one is needed, is the k = 2 from distinguishability —
it is the conservative bound that holds regardless of estimation quality.
Identical thresholds on seeds 7, 123, 2026.

## Measured results — part 3: hard overlap recall (seed 42)

Run: `python scripts/measure_hard.py` — reproduces
`results/hard_results.json`.

The easy recall test (part 1) stored the 4 mutually orthogonal bulk basis
states, so random probes overlapped ~0.12 while noisy probes stayed ~0.99:
the gate was not doing hard work. `tensor_membrane/hard_recall.py` makes
recall genuinely hard and measures it honestly.

**Design.** Non-orthogonal stored patterns:
ψ_j = (√(1−s²)·e_j + s·c)/N, c = uniform (1,1,1,1)/2. s = 0 gives the
orthonormal basis; s = 1 gives all patterns identical. The measured mean
pairwise |overlap| ρ(s) rises monotonically 0 → 1 (s=0.5 → ρ=0.477,
s=0.7 → ρ=0.660, s=0.9 → ρ=0.864, s=0.95 → ρ=0.925). Because V is an
isometry, boundary patterns have exactly the same pairwise overlaps as the
bulk patterns. Three probe types: *noisy* (a stored pattern + noise),
*isotropic random* (a Haar-random 128-dim boundary state), *code-random*
(a Haar-random bulk state encoded to the boundary — it lives in the code
subspace but is NOT a stored pattern; the gate should refuse it). Two
noise models: *boundary noise* (isotropic 128-dim noise) and *subspace
noise* (the bulk state is perturbed before encoding — this moves the state
inside the code subspace, so it can flip the argmax between similar
patterns). Collision probes: a superposition of two stored patterns.

### Finding 1: pattern similarity does NOT break recognition (boundary noise)

Recognition is **1.000 at every (ρ, σ)** tested under isotropic boundary
noise — even ρ=0.925, σ=0.50, with zero misnaming. The similarity knob is
a dud for recognition: the noise is isotropic in 128 dims, so its
projection onto the 4-dim code subspace is tiny (~√(4/128)); the argmax is
decided by the clean overlaps, and ρ < 1 always favours the right pattern.
The failure mode at high boundary noise is MISSING (refusing real
patterns), not misnaming: at σ=1.5, τ=0.6, recognition falls to 0.095 with
miss rate 0.905.

### Finding 2: subspace noise breaks recognition by confident misnaming

Recognition first drops below 1.0 at:

| ρ | breaks at σ_sub | recognition there |
|-----|-----------------|-------------------|
| 0.000 | ~1.0 | 0.838 |
| 0.477 | ~0.7 | 0.950 |
| 0.660 | ~0.5 | 0.990 |
| 0.864 | ~0.5 | 0.805 |
| 0.925 | ~0.3 | 0.910 |

At (ρ=0.864, σ_sub=0.5): recognition 0.805, **misname 0.195, miss
0.000**. At (ρ=0.925, σ_sub=1.0): recognition 0.443, misname 0.545.
The gate almost never refuses a real pattern; it confidently names the
wrong one. That is the failure mode.

### Finding 3: no τ separates real from code-random at high similarity

Best achievable min(recognition, refusal-code) over τ in [0.3, 0.9]:

| case | best min |
|------|----------|
| subspace, ρ=0, σ_sub=0.5 | 0.823 |
| subspace, ρ=0.864, σ_sub=0.5 | **0.805** |
| subspace, ρ=0.864, σ_sub=1.0 | **0.507** |
| subspace, ρ=0.925, σ_sub=1.5 | **0.335** |
| boundary, ρ=0.864, σ=0.5 | 0.907 |
| boundary, ρ=0.864, σ=1.0 | 0.635 |
| boundary, ρ=0.864, σ=1.5 | 0.303 |

At (ρ=0.864, σ_sub=0.5) no threshold gets both recognition and
code-refusal above 0.805; at σ_sub=1.0 it collapses to ~0.5. This is the
membrane-arena shield-strictness result in miniature: **no threshold
separates the classes**. Raising τ trades recognition for refusal (at
ρ=0.9, σ_sub=0.5: τ=0.8 gives recog 0.805 / refuse-code 0.863; τ=0.9 gives
recog 0.570 / refuse-code 0.963).

At the natural τ=0.5 the gate accepts nearly every code-random probe
(refusal 0.000 at ρ=0, 0.363 at ρ=0.864). Only a high τ (~0.9) refuses
them, and only at low noise. The gate refuses isotropic random, not
unfamiliar structure in the code subspace — the same limit membrane-brain
found.

### Finding 4: the gate has no collision detection

A superposition of two stored patterns is **always accepted** (accept rate
1.000 at every ρ) and named as one of the pair, never "other", never
refused. Mean max overlap: 0.707 at ρ=0, 0.859 at ρ=0.477, 0.965 at
ρ=0.864 — the overlap with the *other* two patterns is also high (0.895
at ρ=0.864), so the gate cannot even tell which pair collided. Worse: the
overlap with the two colliding patterns is *exactly equal* by symmetry,
(1+ρ)/√(2+2ρ) — measured differences are ~1e-16, so which one the gate
names is floating-point noise. It has no way to prefer one, and no way to
admit ambiguity.

### Robustness (part 3)

Same qualitative results on network seeds 7 and 123: subspace recognition
at (ρ=0.864, σ_sub=0.5) is below 0.9 with misnaming, refusal-random stays
1.000.

## Tests

`python -m pytest tests/ -q` — **38 passed, 0 failed** (14 base + 10
decoder + 14 hard recall).

- Base: gate embedding is unitary, determinism, layer counts (6/7/6),
  isometry, boundary orthogonality, delocalisation bounds, erasure
  threshold (1 holds, 2 holds, 3 degrades, monotone decay), overlap recall
  (clean, noisy, large noise).
- Decoder: reduced_op vs the original partial_trace, channel matrix
  reproduces the partial trace, no-erasure exact recovery, decoder output
  is always a valid density matrix, full rank up to k=4, fidelity monotone
  in k (exact), determinism, noise hurts more at larger k, conditioning
  grows with k.
- Hard recall: pattern construction (endpoints, monotonicity, bad s),
  encoding preserves overlap, determinism, isotropic noise never breaks
  recognition, code-random accepted at natural τ, subspace noise breaks by
  misnaming, breaking point moves with similarity, no τ separates at high
  similarity, isotropic separation breaks between σ=0.7 and σ=1.0,
  collision always accepted and tied, recognition degrades with
  similarity, robustness on another network seed.

## How to run

```bash
pip install numpy          # the only dependency
python scripts/demo.py                  # part 1: base measurements
python scripts/demo.py --write          # also writes results/results.json
python scripts/demo_decoder.py          # part 2: decoder measurements
python scripts/demo_decoder.py --write  # also writes results/decoder_results.json
python scripts/measure_hard.py          # part 3: hard recall sweeps
python scripts/measure_hard.py --write  # also writes results/hard_results.json
python -m pytest tests/ -q              # 38 tests
```

All three demo scripts are deterministic (seed 42 by default); re-running
them reproduces the numbers in `results/`.

## Honest limits — read before any use

- **Not a real quantum error-correcting code.** The tensors are random
  unitaries, not a designed code (no stabilisers, no syndrome, no decoder
  circuit). Part 1's "erasure recovery" is a distinguishability measure
  (trace distance on survivors), not a demonstrated decoding procedure;
  part 2's decoder is a classical linear inversion on a simulated density
  matrix, not a physical recovery operation.
- **Not a real holographic duality.** There is no AdS/CFT here, no
  Ryu–Takayanagi, no bulk reconstruction. The network is a small classical
  simulation of the *structure* (a layered isometry from bulk to boundary).
- **The wiring is our choice.** The sketch gives layer counts (6/7/6) and
  a fan-in/fan-out shape, not exact qubit pairs. A different pairing would
  give slightly different numbers (the seed sweep shows the qualitative
  results are stable, but the exact values depend on the wiring).
- **The base overlap recall is easy by construction.** The stored patterns
  are mutually orthogonal (the encoding is an isometry) and the boundary
  space is 128-dimensional, so random probes overlap ~0.12 while noisy
  probes stay ~0.99. The gate threshold is not doing hard work there —
  the classes are far apart. Part 3 makes it hard, and the gate fails.
- **The gate fails on hard recall.** No threshold separates real patterns
  from code-random states at high similarity and subspace noise, and the
  gate has no collision detection: a superposition of two stored patterns
  is always accepted and named as one of the pair.
- **The decoder knows V and the erasure locations.** In a real device the
  decoder would need the code structure (known) and the erasure locations
  (known for erasures, unlike general errors). The noise model is a proxy
  for finite-sample tomography error, not a simulation of a specific
  tomography protocol, and the practical threshold depends on it — that
  dependence is the finding.
- **The bulk is only 2 qubits.** A 2-qubit message in a 7-qubit boundary
  is a toy. Nothing here scales to a real device, and no claim is made
  that it does.
- **Random unitaries, not perfect tensors.** A HaPPY-style code would use
  perfect tensors with specific entanglement properties; these are
  Haar-random 2-qubit unitaries. The measured threshold (2) matches the
  Singleton bound, but that is a property of this random instance, not a
  proven code family.

## Files

- `tensor_membrane/model.py` — the model (network, encoding, measurements)
- `tensor_membrane/decoder.py` — the real erasure decoder
- `tensor_membrane/hard_recall.py` — hard overlap recall
- `scripts/demo.py` — part 1 measurements
- `scripts/demo_decoder.py` — part 2 measurements
- `scripts/measure_hard.py` — part 3 sweeps
- `tests/test_model.py` — 14 tests (part 1)
- `tests/test_decoder.py` — 10 tests (part 2)
- `tests/test_hard_recall.py` — 14 tests (part 3)
- `results/` — the measured numbers (seed 42): `results.json`,
  `decoder_results.json`, `hard_results.json`
- `README.md`, `CHANGELOG.md`, `CITATION.cff`, `.zenodo.json`,
  `pyproject.toml`, `LICENSE` (GNU AGPL v3)

## Archive and citation

Archived on Zenodo (DOI pending — to be added after the first release).
Cite via `CITATION.cff` (GitHub's "Cite this repository"). License:
GNU AGPL v3 only. Author: Jean-François Brisson, Spark AI NLP,
Fredericton, New Brunswick, Canada (ORCID 0009-0000-9778-5374).
