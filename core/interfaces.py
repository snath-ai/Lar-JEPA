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

class AbstractLatentFaultLocator(ABC):
    """
    Formal specification of the Topological Vulnerability Targeting Engine.

    This abstract base class defines the mathematical principle — not a specific
    implementation — that constitutes the pre-existing prior art claimed in
    Exhibit A of the Lár-JEPA IP disclosure (Section 3, Cryptographic Hash:
    616b53b5350cab38c634e0414b35512f2381ee7da3f32a6a1a2ae4f3b1d299d4).

    Mathematical Specification
    --------------------------
    LatentFaultLocator(x_E, x_S, k) → (risk_score, coordinates)

    Given:
        x_E ∈ ℝ^(B × N_E × d_E)  —  environmental state observations
                                       (B: batch, N_E: sensor/probe count,
                                        d_E: per-sensor feature dimension)
        x_S ∈ ℝ^(1 × N_S × d_S)  —  structural sequence
                                       (N_S: structural position count,
                                        d_S: per-position feature dimension)
        k   ∈ ℤ₊                  —  number of fault coordinates to extract

    Algorithm (the invariant — survives any implementation change):
        1. Z_E = Encoder_E(x_E)  ∈ ℝ^(B × D)        environmental latent,
                                                       mean-pooled → Query
        2. Z_S = Encoder_S(x_S)  ∈ ℝ^(1 × N_S × D)  structural latent,
                                                       positional → Key, Value
        3. Q   = W_Q(Z_E)        ∈ ℝ^(B × 1 × D)
        4. K   = W_K(Z_S)        ∈ ℝ^(B × N_S × D)  (broadcast across batch)
        5. V   = W_V(Z_S)        ∈ ℝ^(B × N_S × D)
        6. α   = softmax(Q Kᵀ / √D)  ∈ ℝ^(B × 1 × N_S)   attention weights
        7. ctx = α V              ∈ ℝ^(B × D)
        8. s   = σ(W_out(ctx))   ∈ [0, 1]            risk score
        9. C   = topk(α, k)                           structural fault coordinates
                                                       (indices into x_S)

    Invariant properties verified by the behavioral test suite
    (lar_jepa/tests/unit/test_latent_fault_locator_invariants.py):
        I1. encode_environmental_state(x_E).shape == (B, D)          [pooled Query]
        I2. encode_structural_sequence(x_S).shape == (1, N_S, D)     [positional K/V]
        I3. α.sum(dim=-1) ≈ 1.0                                       [valid distribution]
        I4. s ∈ [0.0, 1.0]                                            [valid probability]
        I5. C ⊆ {0, 1, …, N_S − 1}                                   [valid coordinates]
        I6. len(C) == k                                                [correct count]

    Domain Isomorphism
    ------------------
    Any tuple (x_E, x_S) in which x_E represents continuous observations of an
    environmental or system state and x_S represents discrete positions in a
    structural topology is a valid instantiation of this specification.

    Proven reference implementations (all pre-employment, all published open-source
    in this repository under Apache 2.0):

        Materials domain:
            x_E = electrochemical operating conditions (B × N_stations × 12)
            x_S = crystal lattice elemental sites      (1 × N_sites × 6)
            C   = topk crystal positions driving instability
            See: examples/materials_jepa_showcase.py, CycleStabilityHead

        Seismic domain:
            x_E = crustal stress field readings        (B × N_stations × 6)
            x_S = geological fault segment topology    (1 × N_segments × 6)
            C   = topk fault coordinates at highest seismic risk
            See: examples/seismic_jepa_showcase.py, TectonicRiskHead

        Infrastructure domain:
            x_E = network traffic load telemetry       (B × N_probes × 6)
            x_S = server / router graph topology       (1 × N_nodes × 6)
            C   = topk critical network failure nodes
            See: examples/infrastructure_jepa_showcase.py, CriticalNodeHead

        Biomedical / Genomic domain:
            x_E = single-cell RNA-seq disease expression profile  (B × N_genes)
                  (patient's cancer cell transcriptomic state — the environmental
                   context that conditions which DNA positions are viable targets)
            x_S = JEPA-encoded DNA gene sequence                  (1 × SeqLen × D)
                  (structural positional embedding of the candidate gene)
            C   = topk base-pair intervention coordinates
                  (positions receiving highest cross-attention — predicted CRISPR
                   guide-RNA cut sites conditioned on the patient's disease state)
            See: lar_jepa/tests/unit/test_latent_fault_locator_invariants.py,
                 GenomicLatentFaultLocator

            Domain isomorphism:
              x_E (RNA expression) ↔ x_E (electrochemical stress) — continuous
                environmental signal conditioning which structural positions are
                most vulnerable given the current system state.
              x_S (DNA sequence)   ↔ x_S (crystal lattice)        — discrete
                structural topology over which the environmental state is attended
                to localise the highest-risk positions.
              C   (base pairs)     ↔ C   (crystal sites)          — topk
                structural coordinates at highest risk given the environmental state.

            Any implementation mapping (RNA profile, DNA sequence) → intervention
            coordinates via cross-modal attention satisfies this specification and
            is a Derivative Work of this pre-employment prior art.

    Legal significance
    ------------------
    This ABC is the architectural prior art. Any future implementation of the
    Latent Fault Locator — regardless of which specific encoders, attention
    mechanism variants, or prediction head architectures are used — that satisfies
    invariants I1–I6 and extends this class is legally classified as a Derivative
    Work of this pre-employment specification under the Derivative Works Doctrine
    (Irish Copyright and Related Rights Act 2000, s. 23; Swiss CO Art. 332).

    Compliance with invariants I1–I6 is mechanically verifiable by running:
        pytest lar_jepa/tests/unit/test_latent_fault_locator_invariants.py

    The invariant test suite is itself prior art, published in this repository
    pre-employment. An implementation that passes all six invariants against any
    domain-specific (x_E, x_S) pair is provably a derivative of this specification,
    regardless of internal implementation detail.

    Authorship and prior art timestamp
    -----------------------------------
    Specified by: Aadithya Vishnu Sajeev
    First published: May 2026, prior to employment commencement.
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    Anchored by genesis hash: 616b53b5350cab38c634e0414b35512f2381ee7da3f32a6a1a2ae4f3b1d299d4
    (RFC 3161 certified by FreeTSA.org, March 31, 2026 at 20:37:10 UTC)
    """

    @abstractmethod
    def encode_environmental_state(self, x_E: Any) -> Any:
        """
        Encode the environmental state observations into a pooled latent vector.

        This is Encoder_E in the specification. The output MUST be mean-pooled
        (or otherwise aggregated) to a single vector per batch element — it
        serves as the Query in the cross-attention step. The specific encoder
        architecture (MLP, Transformer, CNN, SSM, GNN, or any future architecture)
        is not constrained by this specification. Only the output contract is:

            output.shape == (B, D)

        where B is the batch size and D is the shared embedding dimension.

        Invariant I1: encode_environmental_state(x_E).shape[-1] == D
                      encode_environmental_state(x_E).ndim == 2

        Parameters
        ----------
        x_E : Any
            Environmental state observations. Shape: (B, N_E, d_E) or equivalent.
            Domain examples:
              Materials  — electrochemical operating condition measurements
              Seismic    — per-station crustal stress field readings
              Network    — per-probe network traffic telemetry vectors
              Biomedical — single-cell RNA-seq disease expression profile
                           (patient cancer cell transcriptomic state)
              [Future]   — any continuous environmental monitoring signal

        Returns
        -------
        Any
            Pooled latent embedding Z_E ∈ ℝ^(B × D). The Query.
        """
        pass

    @abstractmethod
    def encode_structural_sequence(self, x_S: Any) -> Any:
        """
        Encode the structural topology into a positional latent sequence.

        This is Encoder_S in the specification. The output MUST preserve
        positional structure — it is NOT pooled. It serves as both the Key
        and Value in the cross-attention step. The specific encoder architecture
        is not constrained. Only the output contract is:

            output.shape == (1, N_S, D)

        where N_S is the number of structural positions and D is the shared
        embedding dimension.

        Invariant I2: encode_structural_sequence(x_S).shape == (1, N_S, D)
                      encode_structural_sequence(x_S).ndim == 3

        Parameters
        ----------
        x_S : Any
            Structural topology data. Shape: (1, N_S, d_S) or equivalent.
            Domain examples:
              Materials  — crystal lattice elemental site parameters
              Seismic    — geological fault segment geometry/kinematics
              Network    — server/router node centrality and load parameters
              Biomedical — JEPA-encoded DNA gene sequence (base-pair resolution)
              [Future]   — any discrete structural topology positions

        Returns
        -------
        Any
            Positional latent sequence Z_S ∈ ℝ^(1 × N_S × D). The Keys and Values.
        """
        pass

    @abstractmethod
    def localize_fault_coordinates(
        self,
        z_environmental: Any,
        z_structural: Any,
        k: int = 3,
    ) -> tuple:
        """
        Apply cross-attention to project the environmental state against the
        structural sequence and extract the topk fault coordinates.

        This is steps 3–9 of the specification. The cross-attention operator
        may be implemented using any valid attention mechanism — scaled dot-
        product, multi-head attention, linear attention, or any future variant —
        as long as invariants I3–I6 are satisfied:

            I3. attention weights α sum to approximately 1.0 per batch element
            I4. risk_score s ∈ [0.0, 1.0]
            I5. returned coordinate indices ⊆ {0, 1, …, N_S − 1}
            I6. len(coordinates) == k

        Parameters
        ----------
        z_environmental : Any
            Pooled environmental latent Z_E ∈ ℝ^(B × D). The Query.
        z_structural : Any
            Positional structural latent Z_S ∈ ℝ^(1 × N_S × D). The Key/Value.
        k : int
            Number of fault coordinates to extract.

        Returns
        -------
        tuple
            (risk_score, fault_coordinates, attention_weights)
            risk_score        : float or tensor in [0, 1]
            fault_coordinates : sequence of k integer indices into x_S
            attention_weights : attention distribution over structural positions
        """
        pass

    # -- Convenience composite method ---------------------------------------

    def locate(self, x_E: Any, x_S: Any, k: int = 3) -> tuple:
        """
        Full pipeline: encode both inputs, apply cross-attention, return results.

        This method provides the complete LatentFaultLocator(x_E, x_S, k)
        computation. Subclasses may override for efficiency but must preserve
        the invariants I1–I6.

        Returns
        -------
        tuple
            (risk_score, fault_coordinates, attention_weights)
        """
        z_E = self.encode_environmental_state(x_E)
        z_S = self.encode_structural_sequence(x_S)
        return self.localize_fault_coordinates(z_E, z_S, k=k)


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
