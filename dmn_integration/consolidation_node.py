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
      Writes a memory to ChromaDB +  the JSON journal.
  recall(query: str, max_memories: int) -> str
      Retrieves semantically relevant memories from ChromaDB.

Prior art reference
-------------------
This bridge establishes the 'Namespace Sovereignty' and 'Derivative Works
Doctrine' for linking JEPA predictive simulation to the DMN memory substrate.
See DMN v3.0 preprint: 'The Dream Loop' (to be published Zenodo, April 2026).
"""

import sys
import os
import json
import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DMN Hippocampus import — path relative to DMN/lar inside this repo
# ---------------------------------------------------------------------------
_DMN_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "DMN", "lar", "src")
)
if _DMN_SRC not in sys.path:
    sys.path.insert(0, _DMN_SRC)

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
    """

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        dreams_path: Optional[str] = None,
    ):
        """
        Initialise the consolidation bridge.

        Parameters
        ----------
        chroma_path : str, optional
            Path to the ChromaDB persistent store. Defaults to the DMN's
            default at DMN/lar/data/chroma_db.
        dreams_path : str, optional
            Path to the DMN JSON journal. Defaults to the DMN's default
            at DMN/lar/memory/dreams.json.
        """
        self._hippocampus: Optional["Hippocampus"] = None

        if _HIPPOCAMPUS_AVAILABLE:
            try:
                self._hippocampus = Hippocampus(
                    chroma_path=chroma_path,
                    dreams_path=dreams_path,
                )
                print("✅ [JEPA→DMN] Hippocampus connection established.")
            except Exception as e:
                print(f"⚠️  [JEPA→DMN] Hippocampus init failed: {e}. "
                      "Running in degraded mode — trajectory heuristics will not be persisted.")
        else:
            print("⚠️  [JEPA→DMN] DMN Hippocampus not importable. "
                  "Running in degraded mode — trajectory heuristics will not be persisted.")

    # ------------------------------------------------------------------

    def write_trajectory_heuristic(
        self,
        trajectory_log: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Consolidate a committed JEPA trajectory into the DMN episodic store.

        Call this after AbstractEntropicRouter returns COMMIT_TRAJECTORY. The
        trajectory_log captures what the JEPA predicted, what route was chosen,
        and what the outcome was. This record becomes retrievable future context.

        Parameters
        ----------
        trajectory_log : dict
            A dictionary describing the committed trajectory. Expected keys:

            - domain       : str   — domain label (e.g. "spatial_kinematics")
            - action       : Any   — the action vector that was committed
            - predicted_state : Any — the JEPA's ŝ for this action
            - entropic_loss  : float — H(ŝ) score at commit time
            - outcome      : str   — "committed" | "replanned" | "impasse"
            - metadata     : dict  — any additional domain-specific context

        embedding : List[float], optional
            Pre-computed embedding vector for the trajectory summary text.
            If not provided, the Hippocampus will attempt to generate one
            via its configured Ollama embedding endpoint.

        Returns
        -------
        bool
            True if the heuristic was successfully written, False otherwise.
        """
        if self._hippocampus is None:
            return False

        # Construct a human-readable summary for semantic search
        domain      = trajectory_log.get("domain", "unknown")
        outcome     = trajectory_log.get("outcome", "unknown")
        entropy     = trajectory_log.get("entropic_loss", -1.0)
        action_repr = str(trajectory_log.get("action", ""))[:120]

        summary = (
            f"[JEPA Heuristic] Domain: {domain} | Outcome: {outcome} | "
            f"Entropic loss: {entropy:.4f} | Action: {action_repr}"
        )

        metadata = {
            "source":        "jepa_consolidation_node",
            "domain":        domain,
            "outcome":       outcome,
            "entropic_loss": entropy,
            "memory_type":   "episodic",          # DMN v3.0 memory typing
            "timestamp":     datetime.datetime.utcnow().isoformat(),
            **trajectory_log.get("metadata", {}),
        }

        try:
            if embedding:
                self._hippocampus.save_memory(summary, embedding, metadata)
            else:
                # Generate embedding via DMN Hippocampus's own embed endpoint
                generated_embedding = self._hippocampus._generate_embedding(summary)
                if generated_embedding:
                    self._hippocampus.save_memory(summary, generated_embedding, metadata)
                else:
                    # Write to JSON journal only (no vector search, but still persisted)
                    self._hippocampus.save_memory(summary, [], metadata)
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

        Call this at the start of a JEPA planning cycle to provide the
        BrainNode or the JEPA encoder with prior successful strategies.

        Parameters
        ----------
        query : str
            A natural-language description of the current planning context
            (e.g. "collision avoidance in dense 3-body orbital field").

        max_results : int
            Maximum number of heuristics to retrieve. Default 3.

        Returns
        -------
        str
            Concatenated heuristic summaries, or empty string if none available.
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

        Retained for backward compatibility with any code referencing the
        original stub interface. New code should call write_trajectory_heuristic
        directly for explicit control over the embedding.
        """
        if isinstance(trajectory_log, dict):
            return self.write_trajectory_heuristic(trajectory_log)
        # If given a non-dict, wrap it and write
        return self.write_trajectory_heuristic({"raw": str(trajectory_log)})
