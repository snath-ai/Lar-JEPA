"""
CognitiveNodeAdapter
====================
Bridges the AbstractCognitiveNode interface to the Lár BaseNode interface,
making any AbstractCognitiveNode implementation directly usable inside a
BatchNode, RouterNode path_map, or any position in the Lár execution graph.

Without this adapter, AbstractCognitiveNode (core/interfaces.py) and BaseNode
(lar_jepa/src/lar/node.py) are two separate class hierarchies. The adapter
closes the gap so that:

    BatchNode([
        JEPAAdapter(my_jepa_node),   # AbstractManifold subclass
        LLMNode(...),                # Standard Lár node
        JEPAAdapter(other_jepa),     # Second JEPA configuration
    ])

...works correctly — each running concurrently in its own thread with an
isolated GraphState copy, with results merged back by BatchNode's fan-in.

Usage
-----
    from core.adapter import CognitiveNodeAdapter

    class MyJEPA(AbstractManifold):
        model_type = ModelType.JEPA
        ...

    jepa_node = MyJEPA()

    # Wrap for use in the Lár graph
    wrapped = CognitiveNodeAdapter(
        cognitive_node=jepa_node,
        input_key="sensor_observation",   # GraphState key to read as input_signal
        output_key="jepa_prediction",      # GraphState key to write decoded output
        next_node=router_node,
    )

    # Drop directly into BatchNode — works alongside LLMNode instances
    batch = BatchNode([wrapped, llm_node, other_jepa_wrapped])
"""

import sys
import os
from typing import Any, Optional

# Resolve the Lár framework src path relative to this file
_LAR_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lar_jepa", "src")
)
if _LAR_SRC not in sys.path:
    sys.path.insert(0, _LAR_SRC)

from lar.node import BaseNode
from lar.state import GraphState
from .interfaces import AbstractCognitiveNode


class CognitiveNodeAdapter(BaseNode):
    """
    Adapts any AbstractCognitiveNode (LLM, JEPA, diffusion, SSM, GNN, etc.)
    to the Lár BaseNode interface so it can be used anywhere a BaseNode is
    accepted — including inside BatchNode for concurrent execution.

    The adapter implements the three-phase AbstractCognitiveNode contract:
        1. encode(input_signal)    — convert GraphState value to internal context
        2. forward(context)        — run the node's inference/prediction pass
        3. decode(representation)  — convert internal output to GraphState value

    This means the wrapped node retains its full AbstractCognitiveNode contract
    and can also be used independently of the Lár graph if needed.

    Thread Safety
    -------------
    BatchNode deep-copies the GraphState before passing it to each thread.
    CognitiveNodeAdapter reads from and writes to its local state copy only.
    The wrapped AbstractCognitiveNode must be stateless at inference time
    (i.e., it should not mutate shared instance variables during execute()).
    This is the standard expectation for all Lár nodes.
    """

    def __init__(
        self,
        cognitive_node: AbstractCognitiveNode,
        input_key: str,
        output_key: str,
        next_node: Optional[BaseNode] = None,
        action_key: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        cognitive_node : AbstractCognitiveNode
            The wrapped cognitive node (any ModelType subclass).

        input_key : str
            The GraphState key to read as the input_signal for encode().
            For a JEPANode: the raw observation field.
            For an LLM wrapper: a text prompt field.

        output_key : str
            The GraphState key to write the decoded output.
            For a JEPANode: the latent prediction (LATENT_EMBEDDING).
            For an LLM wrapper: the generated text.

        next_node : BaseNode, optional
            The next node in the Lár graph after this adapter completes.

        action_key : str, optional
            For AbstractManifold subclasses: the GraphState key containing
            the action vector to pass to predict_target(context, action_vector).
            If None and the node is an AbstractManifold, forward() is called
            with no action (stationary prediction).
        """
        if not isinstance(cognitive_node, AbstractCognitiveNode):
            raise ValueError(
                f"cognitive_node must be an AbstractCognitiveNode instance, "
                f"got {type(cognitive_node).__name__}"
            )
        if not isinstance(input_key, str) or not input_key:
            raise ValueError("input_key must be a non-empty string")
        if not isinstance(output_key, str) or not output_key:
            raise ValueError("output_key must be a non-empty string")
        self._validate_next_node(next_node)

        self.cognitive_node = cognitive_node
        self.input_key = input_key
        self.output_key = output_key
        self.next_node = next_node
        self.action_key = action_key

        # Expose model type for logging and introspection
        self.model_type = getattr(cognitive_node, "model_type", None)
        self.__name__ = (
            f"CognitiveNodeAdapter[{self.model_type}:"
            f"{cognitive_node.__class__.__name__}]"
        )

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        """
        Execute the wrapped AbstractCognitiveNode against the current GraphState.

        Reads input_key from state → encode → forward (or predict_target for
        AbstractManifold) → decode → writes output_key to state.
        """
        node_name = self.cognitive_node.__class__.__name__
        model_type = getattr(self.model_type, "value", str(self.model_type))
        print(f"  [CognitiveNodeAdapter] Executing {node_name} (ModelType: {model_type})")

        # Phase 1: Read input signal from GraphState
        input_signal = state.get(self.input_key)
        if input_signal is None:
            msg = (
                f"[CognitiveNodeAdapter] input_key '{self.input_key}' not found "
                f"in GraphState for {node_name}."
            )
            print(f"  WARNING: {msg}")
            state.set("last_error", msg)
            return self.next_node

        try:
            # Phase 2: encode → forward → decode
            context = self.cognitive_node.encode(input_signal)

            # AbstractManifold nodes use predict_target(context, action_vector)
            from .interfaces import AbstractManifold
            if isinstance(self.cognitive_node, AbstractManifold) and self.action_key:
                action_vector = state.get(self.action_key)
                representation = self.cognitive_node.predict_target(context, action_vector)
            else:
                representation = self.cognitive_node.forward(context)

            output = self.cognitive_node.decode(representation)

            # Phase 3: Write decoded output to GraphState
            state.set(self.output_key, output)

            # Write the output signal type as metadata for downstream ContextBridge
            signal_type = self.cognitive_node.output_signal_type
            state.set(
                f"__signal_type_{self.output_key}",
                signal_type.value if hasattr(signal_type, "value") else str(signal_type),
            )

            print(
                f"  [CognitiveNodeAdapter] {node_name} complete. "
                f"Output written to state['{self.output_key}'] "
                f"(SignalType: {signal_type})"
            )

        except Exception as e:
            msg = f"[CognitiveNodeAdapter] {node_name} failed: {e}"
            print(f"  ERROR: {msg}")
            state.set("last_error", str(e))

        return self.next_node
