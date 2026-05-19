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

    Reference Algorithm (one compliant instantiation; the formal invariants are I1–I6 below):
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


# ---------------------------------------------------------------------------

class AbstractAttentionKernel(ABC):
    """
    Formal specification of the attention mechanism used within cross-modal
    cognitive pipelines in the Lár routing graph.

    This ABC decouples the *mechanism* of attention from the
    AbstractLatentFaultLocator specification. Any mechanism that satisfies
    invariants A1–A6 is a valid kernel for cross-modal fault localisation,
    cross-modal composition, or any other attention-gated routing step.

    Mathematical Specification
    --------------------------
    AttentionKernel(Q, K, V, k) → (attention_weights, topk_indices)

    Given:
        Q  ∈ ℝ^(B × 1 × D)   — query (pooled environmental / source state)
        K  ∈ ℝ^(B × N × D)   — keys  (structural / target sequence positions)
        V  ∈ ℝ^(B × N × D)   — values (structural / target sequence positions)
        k  ∈ ℤ₊               — number of positions to extract

    Algorithm (the invariant — survives any mechanism change):
        1. α = attention_fn(Q, K)  ∈ ℝ^(B × N)   weights over N positions
        2. C = topk(α, k)                          k highest-weighted positions

    Invariant properties verified by test_core_interface_invariants.py:
        A1. attention_weights.shape[-1] == N        [covers all N positions]
        A2. topk_indices ⊆ {0, …, N−1}             [valid position coordinates]
        A3. attention_weights ≥ 0                   [non-negative weights]
        A4. attention_weights.sum(dim=-1) ≈ 1.0    [normalised distribution]
        A5. topk_indices ordered descending by weight
        A6. len(topk_indices) == k                  [correct extraction count]

    Valid mechanism implementations (non-exhaustive):
        ScaledDotProductKernel   — softmax(QKᵀ / √D)          [standard]
        LinearAttentionKernel    — φ(Q)φ(K)ᵀ / normalizer      [O(N) complexity]
        SparseAttentionKernel    — local window or strided       [long sequences]
        CosineAttentionKernel    — softmax(cos_sim(Q, K))
        SSMAttentionKernel       — state-space recurrence kernel [causal]
        HyenaKernel              — implicit long convolution      [sub-quadratic]
        [Any future mechanism satisfying A1–A6]

    Domain-agnosticism:
        The mechanism is fully domain-agnostic. The same LinearAttentionKernel
        attending from RNA expression to DNA base-pair positions also attends
        from seismic stress tensors to fault segment topology — because the
        kernel operates on (Q, K, V) tensors, not on domain semantics.

    Authorship and prior art timestamp
    ------------------------------------
    Specified by: Aadithya Vishnu Sajeev
    First published: May 2026, prior to employment commencement.
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    Anchored by: Zenodo DOIs 10.5281/zenodo.19245328, 10.5281/zenodo.19484646,
                 10.5281/zenodo.19646405
    """

    @abstractmethod
    def compute(
        self,
        query: Any,
        key: Any,
        value: Any,
        k: int,
    ) -> tuple:
        """
        Compute attention weights over the key/value sequence and extract
        the top-k positions.

        Invariants A1–A6 must hold for all valid inputs.

        Parameters
        ----------
        query : Any
            Pooled query vector. Shape: (B, 1, D) or (B, D).
        key : Any
            Key sequence. Shape: (B, N, D).
        value : Any
            Value sequence. Shape: (B, N, D). May be unused by some mechanisms.
        k : int
            Number of positions to extract. Must satisfy k ≤ N.

        Returns
        -------
        tuple
            (attention_weights, topk_indices)
            attention_weights : tensor (B, N)  — normalised distribution
            topk_indices      : tensor (k,)    — top-k indices, ordered descending
        """
        pass


# ---------------------------------------------------------------------------

class AbstractPerturbationOperator(ABC):
    """
    Formal specification of latent-space perturbation and zero-shot
    counterfactual state prediction.

    This ABC formalises the pattern of computing a directional perturbation
    vector in latent space from a (wildtype, mutant) pair and linearly
    applying it to a control state to predict the post-intervention outcome
    — without executing the intervention in the real world.

    Mathematical Specification
    --------------------------
    PerturbationOperator(z_ctrl, x_wt, x_mut, α) → z_pred

    Given:
        x_wt   — wildtype (unperturbed) input in the source domain
        x_mut  — mutant (perturbed) input in the source domain
        z_ctrl ∈ ℝ^(B × D) — control latent state to perturb
        α      ∈ ℝ         — perturbation magnitude scalar

    Algorithm (the invariant — survives encoder architecture changes):
        1. z_wt   = encode_wildtype(x_wt)         ∈ ℝ^(B × D)
        2. z_mut  = encode_mutant(x_mut)           ∈ ℝ^(B × D)
        3. Δ      = z_mut − z_wt                   ∈ ℝ^(B × D)   perturbation vector
        4. z_pred = z_ctrl + α · Δ                 ∈ ℝ^(B × D)   predicted state

    Invariant properties verified by test_core_interface_invariants.py:
        P1. encode_wildtype(x).shape == encode_mutant(x).shape == (B, D)
        P2. perturbation_vector = encode_mutant(x_mut) − encode_wildtype(x_wt)
        P3. predict_perturbed_state(z, wt, mut, α=0) ≈ z    [identity at α=0]
        P4. predict_perturbed_state is linear in α
            (doubling α doubles the displacement from z_ctrl)
        P5. perturbation_vector(x_wt, x_mut) is independent of z_ctrl
        P6. deterministic — same inputs always produce the same perturbation

    Domain instantiations (non-exhaustive, all pre-employment prior art):

        Genomic knockout prediction:
            x_wt   = wildtype gene DNA / RNA sequence
            x_mut  = CRISPR-edited or knocked-out variant sequence
            z_ctrl = patient's current transcriptomic disease state
            z_pred = predicted post-knockout cell transcriptomic state
            α = 1.0 (full knockout)

        Materials defect simulation:
            x_wt   = perfect crystal lattice configuration
            x_mut  = defect-injected crystal (vacancy, dopant, strain field)
            z_ctrl = current electrochemical operating state
            z_pred = predicted stability shift under the defect

        Protein conformation prediction:
            x_wt   = unbound protein structure
            x_mut  = ligand-bound protein conformation
            z_ctrl = cellular environmental context
            z_pred = predicted conformational shift upon ligand binding

        Climate perturbation modelling:
            x_wt   = baseline atmospheric state
            x_mut  = perturbed atmospheric state (elevated CO₂, temperature)
            z_ctrl = current climate trajectory latent
            z_pred = predicted system evolution under the perturbation

        Molecular dynamics:
            x_wt   = ground-state molecular geometry
            x_mut  = excited / transition-state geometry
            z_ctrl = reaction environment state
            z_pred = predicted post-excitation trajectory

        [Any domain in which Δ = f(perturbed) − f(unperturbed) is meaningful]

    Authorship and prior art timestamp
    ------------------------------------
    Specified by: Aadithya Vishnu Sajeev
    First published: May 2026, prior to employment commencement.
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    Anchored by: Zenodo DOIs 10.5281/zenodo.19245328, 10.5281/zenodo.19484646,
                 10.5281/zenodo.19646405
    """

    @abstractmethod
    def encode_wildtype(self, x_wt: Any) -> Any:
        """
        Encode the unperturbed (wildtype / baseline) input into latent space.

        Invariant P1: output.shape == (B, D).

        Parameters
        ----------
        x_wt : Any
            Wildtype input in the source domain.

        Returns
        -------
        Any
            Latent vector z_wt ∈ ℝ^(B × D).
        """
        pass

    @abstractmethod
    def encode_mutant(self, x_mut: Any) -> Any:
        """
        Encode the perturbed (mutant / edited / intervened) input into latent space.

        Invariant P1: output.shape must match encode_wildtype(x).shape == (B, D).

        Parameters
        ----------
        x_mut : Any
            Mutant / perturbed input in the source domain.

        Returns
        -------
        Any
            Latent vector z_mut ∈ ℝ^(B × D).
        """
        pass

    def perturbation_vector(self, x_wt: Any, x_mut: Any) -> Any:
        """
        Compute the perturbation direction vector Δ = encode_mutant(x_mut) − encode_wildtype(x_wt).

        This is the latent-space representation of the intervention — the direction
        and magnitude of the effect in the model's learned latent geometry.

        Invariant P2: this method always computes the additive difference of encodings.
        Invariant P5: result depends only on (x_wt, x_mut), not on any control state.
        Invariant P6: deterministic for same inputs.

        Parameters
        ----------
        x_wt : Any
            Wildtype / baseline input.
        x_mut : Any
            Mutant / perturbed input.

        Returns
        -------
        Any
            Perturbation vector Δ ∈ ℝ^(B × D).
        """
        return self.encode_mutant(x_mut) - self.encode_wildtype(x_wt)

    def predict_perturbed_state(
        self,
        z_ctrl: Any,
        x_wt: Any,
        x_mut: Any,
        alpha: float = 1.0,
    ) -> Any:
        """
        Predict the post-perturbation latent state: z_pred = z_ctrl + α · Δ.

        This is zero-shot intervention prediction. Without executing any physical
        experiment, simulation step, or wet-lab assay, the operator predicts
        where in latent space the system lands after the intervention.

        Invariant P3: at α=0, returns z_ctrl unchanged (no intervention).
        Invariant P4: displacement from z_ctrl scales linearly with α.

        Parameters
        ----------
        z_ctrl : Any
            Current control state ∈ ℝ^(B × D).
        x_wt : Any
            Wildtype / baseline input.
        x_mut : Any
            Mutant / perturbed input.
        alpha : float
            Perturbation magnitude.
            1.0 = full intervention.  0.5 = partial.  0.0 = no change.

        Returns
        -------
        Any
            Predicted post-perturbation state z_pred ∈ ℝ^(B × D).
        """
        delta = self.perturbation_vector(x_wt, x_mut)
        return z_ctrl + alpha * delta


# ---------------------------------------------------------------------------

class AbstractRoutingKernel(ABC):
    """
    Formal specification of the routing decision function within the
    Lár heterogeneous cognitive graph.

    This ABC decouples the routing *logic* from the routing *mechanism*.
    The standard Lár RouterNode evaluates a Python predicate over GraphState.
    AbstractRoutingKernel formalises the score-then-route pattern, enabling
    learned, probabilistic, topological, and adaptive routing strategies
    alongside the current deterministic threshold approach — without modifying
    the graph executor.

    Mathematical Specification
    --------------------------
    RoutingKernel(state) → (score, route_key)

    Given:
        state — current GraphState or any hashable state representation

    Algorithm:
        1. s = score(state)    ∈ ℝ     continuous routing signal
        2. r = route(state)    ∈ str   discrete next-node key

    Invariant properties verified by test_core_interface_invariants.py:
        R1. score(state) returns a finite float
        R2. route(state) returns a non-empty string
        R3. same state always produces same (score, route) — deterministic
        R4. route is consistent with score across independent calls

    Valid routing implementations (non-exhaustive):
        EntropicThresholdKernel   — route on JEPA entropic loss threshold [current]
        ConfidenceThresholdKernel — route on model output confidence
        TopologicalKernel         — route on structural graph properties
        LearnedPolicyKernel       — RL-trained routing policy (score = Q-value)
        EnsembleVoteKernel        — majority vote across heterogeneous agents
        UncertaintyKernel         — route on epistemic uncertainty estimate
        CalibratedBayesianKernel  — posterior predictive routing
        [Any future routing mechanism satisfying R1–R4]

    Authorship and prior art timestamp
    ------------------------------------
    Specified by: Aadithya Vishnu Sajeev
    First published: May 2026, prior to employment commencement.
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    Anchored by: Zenodo DOIs 10.5281/zenodo.19245328, 10.5281/zenodo.19484646,
                 10.5281/zenodo.19646405
    """

    @abstractmethod
    def score(self, state: Any) -> float:
        """
        Compute a continuous routing signal from the current state.

        The signal quantifies whatever drives the routing decision:
        entropy, confidence, uncertainty, Q-value, distance, etc.

        Invariant R1: returns a finite Python float.
        Invariant R3: deterministic — same state, same score.

        Parameters
        ----------
        state : Any
            Current graph state or extracted signal value.

        Returns
        -------
        float
            Continuous routing score. Must be finite.
        """
        pass

    @abstractmethod
    def route(self, state: Any) -> str:
        """
        Return the routing key (next node identifier) for the given state.

        Invariant R2: returns a non-empty string.
        Invariant R3: deterministic — same state, same route.
        Invariant R4: consistent with score — the mapping score→route is stable.

        Parameters
        ----------
        state : Any
            Current graph state or extracted signal value.

        Returns
        -------
        str
            The routing key / next node name. Non-empty.
        """
        pass


# ---------------------------------------------------------------------------

class AbstractModalEncoder(ABC):
    """
    Formal specification of the modality-specific encoder that maps raw
    domain observations into the shared latent space of the Lár cognitive graph.

    This ABC formalises the universal encoding pattern that appears across
    all Lár pipelines: raw domain input → shared latent vector. It separates
    the modality-specific encoding logic from all downstream graph routing,
    attention, and memory operations — enabling plug-and-play encoder
    replacement without modifying any other pipeline component.

    Mathematical Specification
    --------------------------
    ModalEncoder(x) → z

    Given:
        x ∈ source_domain   — raw observation in any modality

    Algorithm:
        z = encode(x)       ∈ ℝ^(B × output_dim)

    Invariant properties verified by test_core_interface_invariants.py:
        M1. encode(x).shape == (B, output_dim)     [correct output shape]
        M2. output_dim is constant across all encode() calls
        M3. encode(x) is deterministic for the same input x

    Domain instantiations (non-exhaustive, all pre-employment prior art):

        Genomic sequence encoder:
            input  = DNA / RNA sequence (one-hot or k-mer tokens)
            output = structural / expression latent (B × D)
            example: DNABERT-2 (117M), CrystalSiteEncoder

        Spectroscopic encoder:
            input  = spectral measurement array (B × wavelengths)
            output = material property latent (B × D)

        Network telemetry encoder:
            input  = per-node traffic measurement vectors (B × N_nodes × features)
            output = network state latent (B × D)

        Electrochemical encoder:
            input  = impedance / cycling measurements (B × timepoints × channels)
            output = battery state latent (B × D)

        Imaging encoder:
            input  = pixel tensor (B × C × H × W)
            output = visual feature latent (B × D)

        Expression profile encoder:
            input  = scRNA-seq count matrix (B × N_genes)
            output = transcriptomic state latent (B × D)

        Seismic sensor encoder:
            input  = per-station stress field measurements (B × N_stations × d)
            output = crustal state latent (B × D)

        [Any modality for which a neural encoder produces (B × D) output]

    Authorship and prior art timestamp
    ------------------------------------
    Specified by: Aadithya Vishnu Sajeev
    First published: May 2026, prior to employment commencement.
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    Anchored by: Zenodo DOIs 10.5281/zenodo.19245328, 10.5281/zenodo.19484646,
                 10.5281/zenodo.19646405
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        The embedding dimensionality D produced by encode().

        Invariant M2: constant — same value returned on every call.
        """
        pass

    @property
    @abstractmethod
    def modality(self) -> str:
        """
        Human-readable name of the modality this encoder handles.

        Examples: "genomic_sequence", "electrochemical", "network_telemetry",
                  "imaging", "seismic", "expression_profile", "spectroscopic"
        """
        pass

    @abstractmethod
    def encode(self, x: Any) -> Any:
        """
        Encode raw domain observations into the shared latent space.

        Invariant M1: output.shape == (B, self.output_dim)
        Invariant M3: deterministic for the same input x (in inference mode)

        Parameters
        ----------
        x : Any
            Raw domain observation. Shape and dtype are modality-specific.

        Returns
        -------
        Any
            Latent vector z ∈ ℝ^(B × output_dim).
        """
        pass


# ---------------------------------------------------------------------------

class AbstractDivergenceRouter(ABC):
    """
    Tenth cognitive ABC in the Lár-JEPA cognitive contract.

    Multi-stream routing primitive. Arbitrates between two independent latent
    streams by measuring their geometric relationship. Does not inspect stream
    content — only confidence scores and divergence between predictions.

    This is the formal specification of the divergence-gated architecture
    introduced in the companion paper. The internal mechanism for computing
    confidence and divergence is not specified by this ABC — any function
    satisfying invariants V1–V6 is compliant.

    Mathematical Specification
    --------------------------
    Given two independent stream encoders A and B:

        z_A, c_A = encode_stream_a(x_A)
        z_B, c_B = encode_stream_b(x_B)
        D        = divergence(z_A, z_B)
        decision = route(c_A, c_B, D)

    Invariant properties (V1–V6):
        V1. encode_stream_a(x).confidence ∈ [0, 1]
        V2. encode_stream_b(x).confidence ∈ [0, 1]
        V3. divergence(z_A, z_B) ≥ 0  for all z_A, z_B
        V4. divergence(z, z) = 0       for all z  (identity invariant)
        V5. route(c_A, c_B, D) is a deterministic pure function —
            identical inputs always produce identical RouteDecision
        V6. route receives only scalars (c_A, c_B, D) — it has no access
            to z_A or z_B; the routing decision is blind to stream content

    V6 is the most important invariant. A route() function that inspects
    stream content is a fusion layer in disguise — it becomes a third model
    that blends the two streams before making a decision. V6 enforces the
    architectural boundary: the RouterNode reads the room, not the case.

    Four Routing Rules
    ------------------
    The route() method is specified by four deterministic rules over
    thresholds τ_high, τ_low, and δ:

        Execute:     c_A ≥ τ_h, c_B ≥ τ_h, D < δ
                     → COMMIT_TRAJECTORY  (both agree)

        Investigate: c_A ≥ τ_h, c_B ≥ τ_h, D ≥ δ
                     → TRIGGER_REPLAN     (both confident, disagree — most
                                           informative case; do not fuse)

        Defer:       exactly one of c_A, c_B ≥ τ_h
                     → COMMIT_TRAJECTORY  (defer to confident stream)

        Halt:        c_A < τ_l, c_B < τ_l
                     → STRUCTURAL_IMPASSE (both uncertain; no reliable signal)

    Relationship to Existing ABCs
    ------------------------------
    AbstractDivergenceRouter is a specialisation of AbstractRoutingKernel
    (R1–R4) in the multi-stream case: the "candidate next states" are the
    outputs of two independent latent encoders, and the scoring function is
    the content-blind divergence gate defined by V1–V6.

    Both stream encoders satisfy AbstractModalEncoder (M1–M3). Stream
    independence is enforced by the AbstractContextBridge contract (pure
    function, no side effects between streams).

    Domain Instantiations (non-exhaustive)
    ----------------------------------------
        Vision-Language:
            stream_a = image latent (JEPA encoder)
            stream_b = text latent  (language encoder)
            signal   = caption contradicts scene

        Medical imaging:
            stream_a = scan latent  (radiology image encoder)
            stream_b = report latent (clinical notes encoder)
            signal   = report inconsistent with image findings

        Autonomous vehicles:
            stream_a = sensor latent (LiDAR / camera)
            stream_b = map / semantic latent
            signal   = road state contradicts map

        Cybersecurity:
            stream_a = syscall / network behaviour latent
            stream_b = auth / policy latent
            signal   = behaviour contradicts declared permissions

    Self-Curating Training Curriculum
    -----------------------------------
    When used as training infrastructure (Lár as Training Infrastructure),
    high-divergence cases are accumulated as D_hard:

        D_hard = {i : Δ_i ≥ δ and r_i = TRIGGER_REPLAN}

    D_hard grows automatically at the model's uncertainty boundary.
    No human labeling required. No manually designed curriculum.
    The routing decisions themselves constitute the curriculum.

    Authorship and prior art timestamp
    ------------------------------------
    Specified by: Aadithya Vishnu Sajeev
    First published: May 2026.
    Repository: github.com/snath-ai/Lar-JEPA (Apache 2.0)
    Anchored by: Zenodo DOI 10.5281/zenodo.20278781
                 (Divergence Is Not Noise: Multi-Stream Routing Without
                  Modal Fusion and the Safety-Learning Equivalence)
    """

    @abstractmethod
    def encode_stream_a(self, x_a: Any) -> tuple[Any, float]:
        """
        V1: Encode Stream A into a latent representation.

        Returns
        -------
        tuple[Any, float]
            (z_a, confidence_a) where:
            - z_a        is the latent tensor for stream A
            - confidence_a ∈ [0, 1]  (V1 invariant: clamped to unit interval)
        """
        pass

    @abstractmethod
    def encode_stream_b(self, x_b: Any) -> tuple[Any, float]:
        """
        V2: Encode Stream B into a latent representation.

        Returns
        -------
        tuple[Any, float]
            (z_b, confidence_b) where:
            - z_b          is the latent tensor for stream B
            - confidence_b ∈ [0, 1]  (V2 invariant: clamped to unit interval)
        """
        pass

    @abstractmethod
    def divergence(self, z_a: Any, z_b: Any) -> float:
        """
        V3–V4: Compute the divergence between two stream latents.

        Invariant V3: divergence(z_a, z_b) ≥ 0 for all z_a, z_b
        Invariant V4: divergence(z, z) = 0    for all z  (identity)

        The internal metric is not specified — cosine distance, normalised
        L2, Jensen–Shannon divergence, or binary prediction disagreement
        are all compliant provided V3–V4 hold.

        Parameters
        ----------
        z_a : Any
            Latent tensor from encode_stream_a.
        z_b : Any
            Latent tensor from encode_stream_b.

        Returns
        -------
        float
            Non-negative divergence scalar.
        """
        pass

    @abstractmethod
    def route(
        self,
        confidence_a: float,
        confidence_b: float,
        divergence: float,
    ) -> RouteDecision:
        """
        V5–V6: Deterministic routing from confidence scalars and divergence only.

        Invariant V5: identical (confidence_a, confidence_b, divergence) inputs
                      always produce identical RouteDecision output.
        Invariant V6: this method receives ONLY scalars (c_A, c_B, D).
                      It has no access to z_a or z_b.
                      The routing decision is blind to stream content.

        Parameters
        ----------
        confidence_a : float
            Confidence score from encode_stream_a. Must be in [0, 1].
        confidence_b : float
            Confidence score from encode_stream_b. Must be in [0, 1].
        divergence : float
            Divergence scalar from divergence(). Must be ≥ 0.

        Returns
        -------
        RouteDecision
            One of COMMIT_TRAJECTORY, TRIGGER_REPLAN, STRUCTURAL_IMPASSE.
        """
        pass
