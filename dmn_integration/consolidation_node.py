"""
JEPA ↔ DMN Consolidation Bridge
================================
This module connects the Lár-JEPA world model execution layer to any
AbstractDMN implementation via the lar-dmn blueprint contract.

Role in the architecture
------------------------
After the Lár routing spine commits a JEPA trajectory via COMMIT_TRAJECTORY,
the successful execution trace is consolidated into the DMN's episodic memory
(Tier 1) so that:

  1. Future JEPA planning cycles can retrieve prior successful heuristics
     as warm context — preventing re-exploration of known-good latent regions.

  2. The domain DMN's consolidate() cycle can, during the next idle pass,
     cluster accumulated D_hard events into Tier 2 failure-class centroids
     and Tier 3 LoRA adapters — completing the Consolidation Loop.

AbstractDMN contract (lar-dmn v2.5.0+)
--------------------------------------
  ingest(event: Any) -> None
      Accepts an event into the Tier 1 episodic queue (non-blocking).
  recall(query: Any, **kwargs) -> Any
      Retrieves context from Tier 2 for the current inference cycle.
  consolidate(**kwargs) -> List[dict]
      Processes Tier-1 events into durable Tier-2/3 artifacts (run overnight
      or on threshold).

Prior art reference
-------------------
Establishes the 'Namespace Sovereignty' link from JEPA predictive simulation
to the DMN memory substrate. See EIM paper (DOI: 10.5281/zenodo.20614051)
for the V7 Difficulty Invariance proof — D_hard geometry persists across
encoder upgrades, making this bridge architecture-independent.
"""

import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# AbstractDMN import — optional; falls back to _InMemoryDMN if not installed
# ---------------------------------------------------------------------------
try:
    from brain.abstract_dmn import AbstractDMN as _AbstractDMN
    _ABSTRACT_DMN_AVAILABLE = True
except ImportError:
    _AbstractDMN = object  # type: ignore
    _ABSTRACT_DMN_AVAILABLE = False


class _InMemoryDMN:
    """
    Minimal in-memory DMN fallback for demos and tests.

    Not HMAC-signed, not persisted — suitable only for local development.
    For production, pass a proper AbstractDMN subclass to
    JEPA_DMN_Consolidation_Node(dmn=...).
    """

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    def ingest(self, event: Any) -> None:
        self._events.append(event if isinstance(event, dict) else {"raw": str(event)})

    def consolidate(self, **kwargs) -> List[dict]:
        return []

    def recall(self, query: Any, **kwargs) -> str:
        query_str = str(query).lower()
        for ev in reversed(self._events):
            summary = ev.get("summary", "")
            if query_str in summary.lower():
                return summary
        return self._events[-1].get("summary", "") if self._events else ""

    def stats(self) -> dict:
        return {"events": len(self._events)}


# ---------------------------------------------------------------------------

class JEPA_DMN_Consolidation_Node:
    """
    Canonical bridge between the Lár-JEPA world model and an AbstractDMN
    implementation.

    This node sits at the boundary between the JEPA execution layer and the
    DMN memory substrate. It is invoked by the routing graph after the
    AbstractEntropicRouter returns COMMIT_TRAJECTORY — meaning the JEPA's
    predicted trajectory has been validated as structurally sound.

    The node performs two operations:

      1. write_trajectory_heuristic(trajectory_log)
         Converts a committed JEPA trajectory into an event dict and
         ingests it into the DMN (Tier 1 episodic queue).

      2. recall_heuristics(query, max_results)
         Retrieves relevant past JEPA heuristics from the DMN's Tier 2
         semantic memory to warm the planning context for the current cycle.

    Both operations degrade gracefully if no DMN is provided — a minimal
    in-memory fallback is used so that JEPA execution is never blocked.

    Usage
    -----
    Wire a concrete AbstractDMN subclass at construction time:

        from my_domain.dmn import MyDomainDMN
        bridge = JEPA_DMN_Consolidation_Node(dmn=MyDomainDMN())

    Or omit for local demos (in-memory, not persisted):

        bridge = JEPA_DMN_Consolidation_Node()

    Install requirement
    -------------------
    ``pip install lar-dmn``  (or ``pip install -e path/to/DMN/lar``)
    so that ``brain.abstract_dmn`` is importable when providing a
    custom AbstractDMN subclass.
    """

    def __init__(self, dmn: Optional[Any] = None) -> None:
        if dmn is not None:
            self._dmn = dmn
            print("✅ [JEPA→DMN] DMN connection established.")
        else:
            self._dmn = _InMemoryDMN()
            print(
                "⚠️  [JEPA→DMN] No AbstractDMN provided. "
                "Using in-memory fallback — heuristics will not be persisted.\n"
                "Wire a concrete AbstractDMN subclass: "
                "JEPA_DMN_Consolidation_Node(dmn=MyDomainDMN())"
            )

    # ------------------------------------------------------------------

    def write_trajectory_heuristic(
        self,
        trajectory_log: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        Consolidate a committed JEPA trajectory into the DMN episodic store.

        Converts the trajectory log into a domain-agnostic event dict and
        calls ``self._dmn.ingest(event)``. The DMN's consolidate() cycle
        will later cluster these into Tier 2 failure-class centroids.
        """
        domain = trajectory_log.get("domain", "unknown")
        outcome = trajectory_log.get("outcome", "unknown")
        entropy = trajectory_log.get("entropic_loss", -1.0)
        action_repr = str(trajectory_log.get("action", ""))[:120]

        event: Dict[str, Any] = {
            "summary": (
                f"[JEPA Heuristic] Domain: {domain} | Outcome: {outcome} | "
                f"Entropic loss: {entropy:.4f} | Action: {action_repr}"
            ),
            "source": "jepa_consolidation_node",
            "domain": domain,
            "outcome": outcome,
            "entropic_loss": entropy,
            "memory_type": "episodic",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **trajectory_log.get("metadata", {}),
        }

        try:
            self._dmn.ingest(event)
            return True
        except Exception as e:
            print(f"❌ [JEPA→DMN] write_trajectory_heuristic failed: {e}")
            return False

    # ------------------------------------------------------------------

    def recall_heuristics(
        self,
        query: str,
        max_results: int = 3,
    ) -> str:
        """
        Retrieve semantically relevant JEPA trajectory heuristics from the
        DMN Tier 2 semantic memory to warm the current planning cycle.
        """
        try:
            result = self._dmn.recall(query, max_results=max_results)
        except TypeError:
            result = self._dmn.recall(query)
        except Exception as e:
            print(f"⚠️  [JEPA→DMN] recall_heuristics failed: {e}")
            return ""
        return result if isinstance(result, str) else str(result) if result else ""

    # ------------------------------------------------------------------

    def extract_heuristic_from_trajectory(self, trajectory_log: Any) -> Any:
        """Legacy entry point — delegates to write_trajectory_heuristic."""
        if isinstance(trajectory_log, dict):
            return self.write_trajectory_heuristic(trajectory_log)
        return self.write_trajectory_heuristic({"raw": str(trajectory_log)})
