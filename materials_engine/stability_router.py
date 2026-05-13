"""
ThermalStabilityRouter
======================
Deterministic safety gate for battery electrolyte candidates.

Implements AbstractEntropicRouter — the exact same interface used by
EntropicVetoRouter in the spatial kinematics engine. The routing logic
is structurally identical; only the domain interpretation changes.

Two veto conditions (both must pass to COMMIT):
  1. thermal_entropy < threshold  — composition is thermally stable
  2. formation_energy < 0 eV/atom — composition is thermodynamically stable

If either fails → TRIGGER_REPLAN: load next candidate composition.
If the search exhausts all candidates → STRUCTURAL_IMPASSE.

This is the materials-science analog of:
  - EntropicVetoRouter vetoing high-collision-entropy orbital trajectories

The Lár spine does not know or care that the entropy score here
represents thermal decomposition probability rather than spatial
collision probability. It routes AbstractEntropicRouter.evaluate_state()
output. That's the entire contract.
"""

import sys
import os
from typing import Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.interfaces import AbstractEntropicRouter
from core.types import RouteDecision, StructuralImpasseError
from .crystal_manifold import LatentCrystalState


class ThermalStabilityRouter(AbstractEntropicRouter):
    """
    Routes a crystal candidate to COMMIT or REPLAN based on two
    thermodynamic stability criteria.

    Parameters
    ----------
    thermal_threshold : float
        Maximum allowed thermal entropy (decomposition probability).
        Default 0.40 — compositions with >40% decomposition probability
        are vetoed as potential thermal runaway risks.

    max_formation_energy : float
        Maximum allowed formation energy in eV/atom.
        Default 0.0 — positive formation energy means the crystal is
        thermodynamically unstable at standard conditions (will not
        form spontaneously from its elements).
    """

    def __init__(
        self,
        thermal_threshold: float = 0.40,
        max_formation_energy: float = 0.0,
    ):
        self.thermal_threshold = thermal_threshold
        self.max_formation_energy = max_formation_energy

    def evaluate_state(self, predicted_state: Any) -> RouteDecision:
        if isinstance(predicted_state, LatentCrystalState):
            entropy = predicted_state.thermal_entropy
            energy  = predicted_state.formation_energy
            label   = predicted_state.composition_label
        elif isinstance(predicted_state, dict):
            entropy = predicted_state.get("thermal_entropy", 1.0)
            energy  = predicted_state.get("formation_energy", 1.0)
            label   = predicted_state.get("composition_label", "unknown")
        else:
            return RouteDecision.TRIGGER_REPLAN

        if entropy > self.thermal_threshold:
            print(
                f"[ThermalStabilityRouter] VETO '{label}': "
                f"thermal_entropy={entropy:.3f} > threshold {self.thermal_threshold:.2f}. "
                f"Thermal runaway risk — replanning."
            )
            return RouteDecision.TRIGGER_REPLAN

        if energy > self.max_formation_energy:
            print(
                f"[ThermalStabilityRouter] VETO '{label}': "
                f"formation_energy={energy:.3f} eV/atom > 0 — "
                f"thermodynamically unstable — replanning."
            )
            return RouteDecision.TRIGGER_REPLAN

        print(
            f"[ThermalStabilityRouter] COMMIT '{label}': "
            f"thermal_entropy={entropy:.3f}, "
            f"formation_energy={energy:.3f} eV/atom — stable."
        )
        return RouteDecision.COMMIT_TRAJECTORY


class CompositionSearchEdge:
    """
    Tracks replanning attempts across composition space.
    The materials-domain analog of ReplanTrajectoryEdge in the
    spatial kinematics engine.

    Raises StructuralImpasseError when the candidate pool is exhausted,
    propagating up to the Lár executor's registered error handler.
    """

    def __init__(self, max_retries: int = 20):
        self.max_retries = max_retries
        self.retry_count = 0
        self.vetoed_ids: list = []

    def advance(
        self,
        failed_state: LatentCrystalState,
        dmn_feedback: Optional[Any] = None,
    ) -> bool:
        self.retry_count += 1
        self.vetoed_ids.append(failed_state.composition_id)

        if self.retry_count >= self.max_retries:
            raise StructuralImpasseError(
                f"Composition search exhausted after {self.max_retries} candidates. "
                f"No thermally stable electrolyte found in pool. "
                f"Vetoed IDs: {self.vetoed_ids}"
            )

        if dmn_feedback:
            print(
                f"[CompositionSearchEdge] Attempt {self.retry_count}: "
                f"DMN feedback available — guided composition search."
            )
        else:
            print(
                f"[CompositionSearchEdge] Attempt {self.retry_count}: "
                f"Advancing to next candidate composition."
            )
        return True
