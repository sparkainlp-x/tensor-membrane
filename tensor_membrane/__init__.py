"""Tensor membrane — Jean's holographic tensor network as a membrane.

The bulk is the message, the boundary is the hologram, the tensors are
the lens. A bulk logical qubit is delocalised across the boundary —
that is the error correction. Same "holographic" as the membrane work
(information stored in a pattern, recoverable by overlap), in the
quantum error-correction language.

Submodules:
  model       — the network (7 qubits, 2 bulk + 5 ancillas, 19 unitaries
                in 3 layers of 6/7/6), the encoding isometry, and the
                base measurements (isometry, delocalisation, erasure
                distinguishability, overlap recall).
  decoder     — a real erasure decoder (least-squares channel inversion):
                reconstruct the bulk state from the surviving boundary
                qubits and measure recovery fidelity.
  hard_recall — hard overlap recall: non-orthogonal stored patterns,
                code-random probes, subspace noise, collision probes.
                The gate fails here, and that failure is measured.

Synthetic, stdlib + NumPy only, seeded, reproducible, AGPL-3.0-only.
Not a real quantum error-correcting code and not a real holographic
duality — a small classical simulation of the structure.
"""

from . import decoder, hard_recall, model
from .model import __version__

__all__ = ["__version__", "model", "decoder", "hard_recall"]
