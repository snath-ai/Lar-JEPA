from abc import ABC, abstractmethod
from typing import Any, List

class AbstractManifold(ABC):
    """
    The mathematical foundation representing a generic latent world-model configuration.
    This defines the sovereign Engine structure, agnostic to semantic data types (e.g., coordinates, genomes).
    """

    @abstractmethod
    def embed_spatial_coordinates(self, raw_telemetry: Any) -> Any:
        """
        Compress physical or topological array into a latent mathematical representation.
        """
        pass

    @abstractmethod
    def predict_future_trajectory(self, current_state: Any, action_vector: List[float]) -> Any:
        """
        Project state transit over step (t+1) within the domain-specific constraints.
        """
        pass

class AbstractEntropicRouter(ABC):
    """
    The Lár deterministic spine. A formal DAG router that evaluates the entropic loss matrix
    of an abstract trajectory state and acts as an absolute deterministic replanning ceiling.
    """

    @abstractmethod
    def evaluate_state(self, predicted_state: Any) -> str:
        """
        Evaluate structural deviation to return the programmatic routing pathway.
        """
        pass
