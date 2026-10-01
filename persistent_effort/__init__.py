"""Persistent-effort adaptation for MYCELIUM.

The package intentionally keeps the expensive acoustic model outside the control
plane.  It learns through durable experiment memory, reference conditioning,
parameter search, pairwise/user feedback, and targeted regeneration before it
recommends any weight update.
"""

from .engine import AceStepClient, PersistentEffortEngine, PersistentStore

__all__ = ["AceStepClient", "PersistentEffortEngine", "PersistentStore"]
__version__ = "0.1.0"
