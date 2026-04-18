from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from .types import RouteDecision, ModelType, SignalType


class AbstractCognitiveNode(ABC):
    """
    The universal base interface for any cognitive component within the Lár
    routing graph. This is the specification of a single 'neuron' in the
    Lár cognitive nervous system.

    Any model type — large language model (LLM), JEPA world model, diffusion
    model, state-space model (SSM), graph neural network (GNN), classical
    deterministic function, or any architecture that follows — implements this
    interface to become a first-class, routable node in the Lár DAG executor.

    The Lár GraphExecutor routes *between* AbstractCognitiveNodes without
    inspecting what is *inside* them. This makes the routing spine model-agnostic
    by construction. The same BatchNode that fans out across N LLMNodes can
    fan out across N JEPANodes, or across a heterogeneous mixture of both.

    Supported composition patterns (declared for reference, not enforced here):
      LLM → JEPA:
          LLMNode generates an LGSL routing instruction specifying which
          JEPANode to invoke and with which action vector.

      JEPA → LLM:
          JEPANode writes its latent embedding to GraphState; LLMNode reads
          it as semantic context for interpretation or action generation.

      Parallel ensemble (homogeneous):
          BatchNode([JEPANode, JEPANode, JEPANode]) spins N identical world
          models concurrently with different initial conditions or action vectors.

      Parallel ensemble (heterogeneous):
          BatchNode([LLMNode, JEPANode, GNNNode]) runs mixed model types
          concurrently; a ReduceNode or RouterNode aggregates their outputs.

      Cross-modal cross-attention:
          AbstractContextBridge adapts one node's output SignalType to
          another node's expected encode() input, enabling LLM semantic
          embeddings to condition a JEPA context encoder, and vice versa.

      Hierarchical routing:
          A RouterNode inspects GraphState and selects between LLMNode and
          JEPANode (or any two AbstractCognitiveNode subclasses) based on
          task type determined at runtime.
    """

    #: Subclasses declare their model type for routing-layer introspection.
    model_type: ModelType

    @abstractmethod
    def encode(self, input_signal: Any) -> Any:
        """
        Encode the incoming signal into this node's internal representation.

        For LLMs      : tokenise and embed text into a context representation.
        For JEPAs     : encode a raw observation into a latent context vector.
        For diffusion : encode a conditioning signal.
        For SSMs      : encode an input into the state-space initial condition.
        For classical : parse and validate the input for deterministic processing.

        Parameters
        ----------
        input_signal : Any
            The raw signal from GraphState. May be text, a tensor, a structured
            dict, a graph topology, or any format the node supports.

        Returns
        -------
        Any
            This node's internal context representation.
        """
        pass

    @abstractmethod
    def forward(self, context: Any) -> Any:
        """
        Execute this node's primary inference or prediction pass.

        For LLMs      : autoregressive generation given the encoded context.
        For JEPAs     : predict the next latent state given context + action.
        For diffusion : a denoising forward pass.
        For SSMs      : one step of state-space evolution.
        For classical : deterministic computation over prepared context.

        Parameters
        ----------
        context : Any
            The encoded internal representation produced by encode().

        Returns
        -------
        Any
            The node's internal output representation (pre-decode).
        """
        pass

    @abstractmethod
    def decode(self, representation: Any) -> Any:
        """
        Decode this node's internal output representation into the signal
        written to GraphState.

        For LLMs      : detokenise to text or structured output.
        For JEPAs     : optionally decode latent to observable space. If not
                        required, return the latent representation directly.
        For diffusion : produce the denoised output sample.
        For classical : return the computed result.

        Parameters
        ----------
        representation : Any
            The internal output produced by forward().

        Returns
        -------
        Any
            The output signal written to GraphState for downstream nodes.
        """
        pass

    @property
    @abstractmethod
    def output_signal_type(self) -> SignalType:
        """
        Declares the SignalType of the value this node writes to GraphState.
        Used by AbstractContextBridge to validate cross-modal connections at
        graph construction time.

        Returns
        -------
        SignalType
            e.g. SignalType.TEXT for LLMs, SignalType.LATENT_EMBEDDING for JEPAs.
        """
        pass


# ---------------------------------------------------------------------------

class AbstractManifold(AbstractCognitiveNode):
    """
    A JEPA-specific AbstractCognitiveNode for continuous latent-space world models.

    Specialises AbstractCognitiveNode for spatial and temporal manifold
    prediction. The 'manifold' abstraction allows the JEPA to model topological
    deformation, trajectory dynamics, and conformational shifts in a continuous
    latent space without decoding to raw coordinates at every prediction step.

    Subclasses implement the specific JEPA context encoder (f_x), target
    encoder (f_y), predictor (g), and entropic validity function for a given
    domain. The Domain Isomorphism established in the Spatial Kinematics
    Lár-JEPA paper (DOI: 10.5281/zenodo.19484646) guarantees that the same
    tensor operations detecting a spatial collision are isomorphic to those
    detecting a conformational clash — the manifold abstraction is
    domain-agnostic.

    AbstractManifold satisfies the AbstractCognitiveNode interface as follows:
        encode()  → embed_context()
        forward() → predict_target(context, action_vector=None) [stationary]
        decode()  → returns the latent representation directly by default.
    """

    model_type = ModelType.JEPA

    @abstractmethod
    def embed_context(self, raw_observation: Any) -> Any:
        """
        Encode a raw domain observation into a latent manifold context vector.

        This is the JEPA context encoder f_x. The raw observation may be:
        coordinate arrays, a molecular graph, an image tensor, an LLM semantic
        embedding, a financial time-series, or any domain-specific input. The
        method returns the context latent s that conditions the predictor.

        Parameters
        ----------
        raw_observation : Any
            Domain-specific observation data.

        Returns
        -------
        Any
            The latent context vector s = f_x(o).
        """
        pass

    @abstractmethod
    def predict_target(self, context: Any, action_vector: Any) -> Any:
        """
        Predict the latent representation of the next manifold state given
        the current context latent and a proposed action or perturbation vector.

        This is the JEPA predictor g: (s, a) → ŝ, where ŝ is the predicted
        latent of the target state. The predicted latent is evaluated by
        entropic_loss() before any state transition is committed.

        Parameters
        ----------
        context : Any
            The current latent context vector s = f_x(o).
        action_vector : Any
            The proposed action or perturbation. May be None for a stationary
            (zero-action) prediction.

        Returns
        -------
        Any
            The predicted target latent ŝ = g(s, a).
        """
        pass

    @abstractmethod
    def entropic_loss(self, predicted_state: Any) -> float:
        """
        Compute the entropic deviation of a predicted manifold state from
        the valid structural manifold. Used by AbstractEntropicRouter to
        determine COMMIT_TRAJECTORY vs TRIGGER_REPLAN.

        Values above the domain-configured threshold τ trigger replanning.
        This is the mathematical ceiling that makes the nervous system safe:
        the JEPA can imagine freely in latent space, but commits only when
        the predicted state is structurally valid.

        Parameters
        ----------
        predicted_state : Any
            The predicted latent ŝ produced by predict_target().

        Returns
        -------
        float
            The entropic deviation score H(ŝ). Higher = more structurally invalid.
        """
        pass

    # -- AbstractCognitiveNode interface implementation ---------------------

    def encode(self, input_signal: Any) -> Any:
        """Delegates to embed_context for JEPA manifold nodes."""
        return self.embed_context(input_signal)

    def forward(self, context: Any) -> Any:
        """Default forward is a stationary (zero-action) prediction."""
        return self.predict_target(context, action_vector=None)

    def decode(self, representation: Any) -> Any:
        """JEPAs return the latent representation directly by default."""
        return representation

    @property
    def output_signal_type(self) -> SignalType:
        return SignalType.LATENT_EMBEDDING


# ---------------------------------------------------------------------------

class AbstractContextBridge(ABC):
    """
    A signal conduit enabling cross-modal composition within the Lár routing
    graph. An AbstractContextBridge adapts the output signal of one
    AbstractCognitiveNode into the expected input format of another — enabling
    heterogeneous model composition without requiring nodes to know each other's
    internal representations.

    This is the mechanism by which the nervous system enables:

      LLMs attending to JEPA latent predictions:
          The bridge converts a LATENT_EMBEDDING signal to a text-serialised
          or prefix-embedding format that the LLM's encode() can consume.
          The JEPA world model's predicted future latent becomes visible as
          semantic context to the LLM — without the LLM needing to know how
          the latent was produced.

      JEPAs conditioning on LLM semantic embeddings:
          The bridge converts a TEXT or GRAPH_STATE signal to a manifold-
          compatible context vector that the JEPA's embed_context() can encode.
          This allows the LLM's semantic interpretation of a goal to condition
          the JEPA's trajectory prediction.

      Any future cross-modal pattern:
          A new bridge implementation declares its source and target SignalTypes
          and implements the conversion. No changes to existing nodes required.

    Bridges are stateless by design. They hold no model weights. They are
    pure signal format adapters — the synaptic connectors of the nervous system.
    """

    @property
    @abstractmethod
    def source_signal_type(self) -> SignalType:
        """The SignalType this bridge accepts as input."""
        pass

    @property
    @abstractmethod
    def target_signal_type(self) -> SignalType:
        """The SignalType this bridge produces as output."""
        pass

    @abstractmethod
    def bridge(
        self,
        source_output: Any,
        target_node_type: Optional[Type[AbstractCognitiveNode]] = None,
    ) -> Any:
        """
        Transform the source node's output signal into the target node's
        expected encode() input format.

        Parameters
        ----------
        source_output : Any
            The raw output written to GraphState by the source CognitiveNode.
            Type corresponds to source_signal_type (e.g. a torch.Tensor for
            LATENT_EMBEDDING, a str for TEXT).
        target_node_type : Type[AbstractCognitiveNode], optional
            The class of the target AbstractCognitiveNode. Bridge implementations
            may use this to tailor the conversion (e.g., truncation for context-
            window-limited LLMs vs. full precision for JEPA encoders).

        Returns
        -------
        Any
            The adapted signal, ready for the target node's encode() method.
        """
        pass


# ---------------------------------------------------------------------------

class AbstractEntropicRouter(ABC):
    """
    The Lár deterministic replanning spine for JEPA world model outputs.

    Specialises the standard Lár RouterNode for AbstractManifold predicted
    states. Evaluates the entropic deviation of a predicted latent state and
    returns a deterministic RouteDecision that gates whether the trajectory
    is committed or replanned.

    Note: This is a RouterNode specialised for JEPA outputs. For general
    model-type routing (choosing between LLMNode and JEPANode based on task
    type, for example), use the standard Lár RouterNode with a Python
    decision function that reads GraphState directly.
    """

    @abstractmethod
    def evaluate_state(self, predicted_state: Any) -> RouteDecision:
        """
        Evaluate structural deviation of a predicted manifold state and
        return the programmatic routing pathway.

        Parameters
        ----------
        predicted_state : Any
            The predicted latent ŝ output by an AbstractManifold.predict_target().

        Returns
        -------
        RouteDecision
            COMMIT_TRAJECTORY   — state is valid; proceed.
            TRIGGER_REPLAN      — state exceeds entropy threshold; regenerate.
            STRUCTURAL_IMPASSE  — no valid trajectory found; raise exception.
        """
        pass
