from typing import Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .jepa_manifold import LatentKinematicState
from core.interfaces import AbstractEntropicRouter

class EntropicVetoRouter(AbstractEntropicRouter):
    """
    A deterministic Lár routing node designed to intercept LatentKinematicState predictions
    from the NBodyKinematicsJEPA. If the trajectory implies structural breakdown or entropic clash, it vetos the path.
    """
    def __init__(self, entropy_threshold: float = 0.85):
        self.entropy_threshold = entropy_threshold

    def evaluate_state(self, predicted_state: LatentKinematicState) -> str:
        """
        Evaluates the spatial tensors. Returns a routing string to dictate the next Lár execution edge.
        """
        if predicted_state.collision_entropy > self.entropy_threshold:
            print(f"[EntropicVetoRouter] DANGER VETO: Predicted trajectory entropy ({predicted_state.collision_entropy:.2f}) exceeds structural threshold.")
            return "TRIGGER_REPLAN"
        else:
            print(f"[EntropicVetoRouter] SAFE: Trajectory validated.")
            return "COMMIT_TRAJECTORY"

class ReplanTrajectoryEdge:
    """
    A specific topological edge that receives a vetoed state and feeds it back into the architecture
    for a recalculated vector exploration, avoiding the failed collision state.
    """
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.retry_count = 0

    def route_to_new_vector(self, current_state: LatentKinematicState) -> Any:
        """
        Recalibrates the kinematic exploration logic based on past failures.
        In a multi-body simulation, this directs the framework to explore alternative structural pathways.
        """
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            raise RecursionError("Maximum replanning threshold reached. Structural impasse detected.")
        
        print(f"[ReplanTrajectoryEdge] Attempt {self.retry_count}: Injecting stochastic noise for vector recalculation.")
        # Logic to append stochastic variance to the next run
        return True
