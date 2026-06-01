"""
JEPA ↔ DMN Consolidation Bridge
================================
This module connects the Lár-JEPA world model execution layer to the
Default Mode Network (DMN) episodic memory substrate.

Role in the architecture
------------------------
After the Lár routing spine commits a JEPA trajectory via COMMIT_TRAJECTORY,
the successful execution trace should be consolidated into the DMN's
episodic memory (Hippocampus / ChromaDB) so that:

  1. Future JEPA planning cycles can retrieve prior successful heuristics
     as warm context — preventing the JEPA from re-exploring known-good
     latent regions unnecessarily.

  2. The Dreamer consolidation daemon (running in the DMN) can, during
     the next idle cycle, synthesise multiple JEPA trajectory heuristics
     into a semantic narrative — completing one pass of the Consolidation
     Loop (Episodic → Semantic).

  3. Over time, the BrainNode LoRA adaptor can be trained on confirmed
     JEPA routing decisions — completing the Consolidation Loop to the
     procedural (weight-level) tier.

DMN Hippocampus interface
-------------------------
  save_memory(text: str, embedding: List[float], metadata: dict)
      Writes a memory to ChromaDB + the JSON journal.
  recall(query: str, max_memories: int) -> str
      Retrieves semantically relevant memories from ChromaDB.

Prior art reference
-------------------
This bridge establishes the 'Namespace Sovereignty' and 'Derivative Works
Doctrine' for linking JEPA predictive simulation to the DMN memory substrate.
See DMN v3.0 preprint: 'The Dream Loop' (to be published Zenodo, April 2026).
"""

import json
import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DMN Hippocampus import — lar-dmn must be installed (pip install -e ../DMN/lar)
# ---------------------------------------------------------------------------
try:
    from brain.hippocampus import Hippocampus
    _HIPPOCAMPUS_AVAILABLE = True
except ImportError:
    _HIPPOCAMPUS_AVAILABLE = False
    Hippocampus = None  # type: ignore


# ---------------------------------------------------------------------------

class JEPA_DMN_Consolidation_Node:
    """
    The canonical bridge between the Lár-JEPA world model and the DMN
    episodic memory store (Hippocampus / ChromaDB).

    This node sits at the boundary between the JEPA execution layer and the
    DMN memory substrate. It is invoked by the routing graph after the
    AbstractEntropicRouter returns COMMIT_TRAJECTORY — meaning the JEPA's
    predicted trajectory has been validated as structurally sound.

    The node performs two operations:

      1. write_trajectory_heuristic(trajectory_log)
         Converts a committed JEPA trajectory into a text + metadata record
         and writes it to the DMN Hippocampus (ChromaDB + JSON journal).
         This is the Episodic tier write in the Consolidation Loop.

      2. recall_heuristics(query, max_results)
         Retrieves semantically relevant past JEPA heuristics from the DMN
         Hippocampus to warm the planning context for the current cycle.
         This is the Episodic tier read in the Consolidation Loop.

    Both operations degrade gracefully if the DMN Hippocampus is unavailable
    (e.g., ChromaDB not initialised, running in test mode). The JEPA
    execution spine is never blocked by DMN availability.

    Install requirement
    -------------------
    Run ``pip install -e ../DMN/lar`` (or ``poetry install`` in the project
    root after wiring the lar-dmn path dep) so that ``brain`` is importable.
    """

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        dreams_path: Optional[str] = None,
    ):
        self._hippocampus: Optional["Hippocampus"] = None

        if _HIPPOCAMPUS_AVAILABLE:
            try:
                self._hippocampus = Hippocampus(
                    chroma_path=chroma_path,
                    dreams_path=dreams_path,
                )
                print("✅ [JEPA→DMN] Hippocampus connection established.")
            except Exception as e:
                print(
                    f"⚠️  [JEPA→DMN] Hippocampus init failed: {e}. "
                    "Running in degraded mode — trajectory heuristics will not be persisted."
                )
        else:
            print(
                "⚠️  [JEPA→DMN] lar-dmn not installed. "
                "Run: pip install -e ../DMN/lar\n"
                "Running in degraded mode — trajectory heuristics will not be persisted."
            )

    # ------------------------------------------------------------------

    def write_trajectory_heuristic(
        self,
        trajectory_log: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Consolidate a committed JEPA trajectory into the DMN episodic store.
        """
        if self._hippocampus is None:
            return False

        domain      = trajectory_log.get("domain", "unknown")
        outcome     = trajectory_log.get("outcome", "unknown")
        entropy     = trajectory_log.get("entropic_loss", -1.0)
        action_repr = str(trajectory_log.get("action", ""))[:120]

        summary = (
            f"[JEPA Heuristic] Domain: {domain} | Outcome: {outcome} | "
            f"Entropic loss: {entropy:.4f} | Action: {action_repr}"
        )

        raw_meta = {
            "source":        "jepa_consolidation_node",
            "domain":        domain,
            "outcome":       outcome,
            "entropic_loss": entropy,
            "memory_type":   "episodic",
            "timestamp":     datetime.datetime.utcnow().isoformat(),
            **trajectory_log.get("metadata", {}),
        }
        metadata = {
            k: json.dumps(v) if isinstance(v, (list, dict)) else v
            for k, v in raw_meta.items()
        }

        try:
            if embedding:
                self._hippocampus.save_memory(summary, embedding, metadata)
            else:
                # Use public method if available, fallback to protected for backwards compatibility
                gen_fn = getattr(self._hippocampus, "generate_embedding", getattr(self._hippocampus, "_generate_embedding", None))
                generated_embedding = gen_fn(summary) if gen_fn else None
                if generated_embedding:
                    self._hippocampus.save_memory(summary, generated_embedding, metadata)
                else:
                    print("⚠️  [JEPA→DMN] Embedding generation failed. Skipping memory save to avoid ChromaDB dimension mismatch.")
                    return False
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
        DMN episodic store to warm the current planning cycle.
        """
        if self._hippocampus is None:
            return ""

        try:
            return self._hippocampus.recall(query=query, max_memories=max_results)
        except Exception as e:
            print(f"⚠️  [JEPA→DMN] recall_heuristics failed: {e}")
            return ""

    # ------------------------------------------------------------------

    def extract_heuristic_from_trajectory(self, trajectory_log: Any) -> Any:
        """
        Legacy entry point — delegates to write_trajectory_heuristic.
        Retained for backward compatibility.
        """
        if isinstance(trajectory_log, dict):
            return self.write_trajectory_heuristic(trajectory_log)
        return self.write_trajectory_heuristic({"raw": str(trajectory_log)})
