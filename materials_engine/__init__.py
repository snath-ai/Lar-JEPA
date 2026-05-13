"""
Materials-JEPA: Battery Electrolyte Discovery Engine
=====================================================
A domain-specific JEPA world model stack for solid-state and liquid
electrolyte candidate screening, built on the Lár AbstractManifold interface.

Demonstrates domain isomorphism: the same AbstractManifold contract that
routes N-body orbital mechanics routes crystal structure prediction —
without any modification to the Lár execution spine.

Components
----------
CrystalStructureJEPA     — encodes static crystal composition into per-site
                           latent embeddings. Pre-cached once per candidate.
ElectrochemicalJEPA      — encodes experimental operating conditions into
                           a latent profile. Runs live per experiment.
ThermalStabilityRouter   — deterministic safety gate: vetoes thermally
                           unstable compositions before any lab synthesis.
CompositionSearchEdge    — tracks replanning attempts across composition space.
"""
from .crystal_manifold import CrystalStructureJEPA, LatentCrystalState
from .electrochemical_manifold import ElectrochemicalJEPA, LatentElectrochemicalState
from .stability_router import ThermalStabilityRouter, CompositionSearchEdge

__all__ = [
    "CrystalStructureJEPA",
    "LatentCrystalState",
    "ElectrochemicalJEPA",
    "LatentElectrochemicalState",
    "ThermalStabilityRouter",
    "CompositionSearchEdge",
]
