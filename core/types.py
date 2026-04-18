from enum import Enum


class RouteDecision(Enum):
    """
    Deterministic routing edge definitions for the Lár DAG architecture.
    Returned by AbstractEntropicRouter.evaluate_state() to gate JEPA
    trajectory commits.
    """
    COMMIT_TRAJECTORY   = "COMMIT_TRAJECTORY"
    TRIGGER_REPLAN      = "TRIGGER_REPLAN"
    STRUCTURAL_IMPASSE  = "STRUCTURAL_IMPASSE"


class ModelType(Enum):
    """
    First-class model type taxonomy for the Lár cognitive routing graph.

    All values are equally routable as nodes within the same deterministic
    Lár execution spine. The spine does not privilege any model type — LLMs,
    JEPAs, and all future architectures are first-class citizens.

    A BatchNode can be instantiated over any combination of these model types,
    running them concurrently within isolated GraphState copies.
    """
    LLM        = "LLM"        # Large Language Model (any provider via LiteLLM)
    JEPA       = "JEPA"       # Joint-Embedding Predictive Architecture
    DIFFUSION  = "DIFFUSION"  # Diffusion / score-matching model
    SSM        = "SSM"        # State Space Model (Mamba, S4, RWKV, etc.)
    GNN        = "GNN"        # Graph Neural Network
    CLASSICAL  = "CLASSICAL"  # Deterministic non-neural function or tool
    HYBRID     = "HYBRID"     # Heterogeneous composite (e.g., LLM + JEPA)
    FUTURE     = "FUTURE"     # Forward-compatible placeholder for architectures
                              # not yet named or invented.


class SignalType(Enum):
    """
    The type of signal flowing between AbstractCognitiveNode instances in
    the Lár routing graph.

    Each AbstractCognitiveNode declares its output_signal_type. AbstractContextBridge
    implementations use source_signal_type and target_signal_type to validate
    that a cross-modal connection is well-formed at graph construction time.

    The Lár GraphExecutor carries all signal types natively through GraphState
    without conversion — conversion is the responsibility of the bridge layer.
    """
    TEXT               = "TEXT"               # Natural language string (LLM output)
    LATENT_EMBEDDING   = "LATENT_EMBEDDING"   # Dense float tensor (JEPA latent)
    GRAPH_STATE        = "GRAPH_STATE"        # Structured Lár GraphState payload
    LGSL_SPEC          = "LGSL_SPEC"          # Typed LGSL routing instruction
    STRUCTURED_DATA    = "STRUCTURED_DATA"    # JSON / tabular / schema-typed data
    TENSOR             = "TENSOR"             # Raw numerical tensor (untyped)
    IMAGE              = "IMAGE"              # Pixel tensor (vision models)
    AUDIO              = "AUDIO"              # Waveform or spectrogram tensor
    GRAPH              = "GRAPH"              # Graph topology (nodes + edges)
    DISTRIBUTION       = "DISTRIBUTION"       # Probability distribution parameters


class CompositionPattern(Enum):
    """
    Named cross-modal composition patterns supported by the Lár routing graph.

    These are design patterns for documentation and introspection — not runtime
    constraints. They label the intended dataflow topology of a graph section
    to make the architecture self-describing.

    Any combination not listed here is also valid: Lár enforces no constraint
    on which AbstractCognitiveNode types may be connected. The nervous system
    routes any signal to any node; correctness of the signal format is the
    responsibility of the AbstractContextBridge layer.
    """
    LLM_ROUTES_JEPA          = "LLM_ROUTES_JEPA"
    # LLM generates LGSL routing instruction; JEPA executes the specified prediction.

    JEPA_INFORMS_LLM         = "JEPA_INFORMS_LLM"
    # JEPA latent embedding is bridged as context for LLM generation.

    PARALLEL_HOMOGENEOUS     = "PARALLEL_HOMOGENEOUS"
    # BatchNode over N nodes of the same ModelType (e.g. N JEPAs).

    PARALLEL_HETEROGENEOUS   = "PARALLEL_HETEROGENEOUS"
    # BatchNode over nodes of mixed ModelTypes (e.g. LLM + JEPA + GNN).

    HIERARCHICAL             = "HIERARCHICAL"
    # RouterNode selects ModelType per task based on GraphState at runtime.

    CROSS_ATTENTION          = "CROSS_ATTENTION"
    # AbstractContextBridge-mediated cross-modal attention between two nodes.

    SEQUENTIAL_PIPELINE      = "SEQUENTIAL_PIPELINE"
    # Linear chain of mixed ModelType nodes — each output is next node's input.

    RECURSIVE_SELF_IMPROVEMENT = "RECURSIVE_SELF_IMPROVEMENT"
    # AutoResearchNode loop: JEPA/LLM outputs feed back into planning cycle.


class StructuralImpasseError(Exception):
    """
    Raised when the AbstractEntropicRouter exhausts all valid stochastic
    trajectory alternatives, indicating a fundamental architectural or
    kinematic impasse. The Lár graph catches this and routes to the
    registered error_node if one is configured.
    """
    pass
