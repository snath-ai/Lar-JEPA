"""
LÁR WORLD MODEL (JEPA) PROOF OF CONCEPT 
---------------------------------------------------------
This file proves that the Lár Execution Engine is capable of 
orchestrating predictive JEPAs (Joint Embedding Predictive Architectures)
using abstract numerical state representation instead of just text LLMs.

Scenario: Safe Autonomous Pathfinding
An agent must predict if moving FORWARD will cause a crash (Hit Wall),
without executing the move in the real world. It uses a "Model of the World"
to hallucinate futures and pick the safest option.
"""
from lar import BaseNode, RouterNode, GraphExecutor, AddValueNode
import json
import random

# ==========================================
# 1. The World Model (JEPA Mock)
# ==========================================
class PredictiveJepaNode(BaseNode):
    """
    Simulates a JEPA. It takes the current abstract STATE, 
    applies a proposed ACTION, and outputs the PREDICTED STATE.
    Notice there is zero English text generation here. It is pure math/logic computation.
    """
    def __init__(self, action_to_simulate: str, next_node=None):
        self.next_node = next_node
        self.action_to_simulate = action_to_simulate

    def execute(self, state):
        # The abstract state is a vector: [X_Position, Velocity]
        current_x = state.get("x_pos", 0)
        current_v = state.get("velocity", 0)
        
        print(f"  [JEPA] Current World State: [X={current_x}, V={current_v}]")
        print(f"  [JEPA] Simulating Action: '{self.action_to_simulate}'...")
        
        # Simulate physics/world rules conceptually
        if self.action_to_simulate == "ACCELERATE":
             predicted_v = current_v + 5
             predicted_x = current_x + predicted_v
             
        elif self.action_to_simulate == "BRAKE":
             predicted_v = max(0, current_v - 5)
             predicted_x = current_x + predicted_v
             
        # Output the hallucinatory abstract "Future State"
        predicted_state_tensor = {"predicted_x": predicted_x, "predicted_v": predicted_v}
        print(f"  [JEPA] Predicted Future State: {predicted_state_tensor}")
        
        state.set("future_sim", predicted_state_tensor)
        return self.next_node


# ==========================================
# 2. System 2 Evaluator (Safety Router)
# ==========================================
def evaluate_future_danger(state):
    """
    System 2 evaluates the JEPA's simulation to decide if reality is safe to execute.
    """
    future = state.get("future_sim", {})
    predicted_x = future.get("predicted_x", 0)
    
    # The "Wall" is at X = 10
    if predicted_x >= 10:
        print(f"  [System 2 Router] CRASH DETECTED in simulation (Hit Wall at X=10). Vetoing action.")
        return "REPLAN_NODE"
    else:
        print(f"  [System 2 Router] Simulation Safe. Action Approved.")
        return "EXECUTE_NODE"


# ==========================================
# 3. Physical Actuation & Re-planning
# ==========================================
class PhysicalMotorNode(BaseNode):
    def __init__(self, next_node=None):
        self.next_node = next_node
        
    def execute(self, state):
        future = state.get("future_sim")
        print(f"  [MOTOR] Executing physical movement! Applying simulated state to reality.")
        state.set("x_pos", future["predicted_x"])
        state.set("velocity", future["predicted_v"])
        state.set("action_status", "MOVEMENT_SUCCESS")
        return self.next_node

# Replan simply triggers a "BRAKE" simulation instead
replan_simulation = PredictiveJepaNode(action_to_simulate="BRAKE")

# ==========================================
# 4. Constructing the Deterministic Graph
# ==========================================

# End points
task_complete = AddValueNode(key="task_status", value="ARRIVED", next_node=None)
physical_execution = PhysicalMotorNode(next_node=task_complete)

# The Evaluator routes between physical action or going back to the drawing board
safety_router = RouterNode(
    decision_function=evaluate_future_danger,
    path_map={
        "EXECUTE_NODE": physical_execution,
        "REPLAN_NODE": replan_simulation
    },
    default_node=physical_execution
)

# Link the replanner back into the safety review loop 
replan_simulation.next_node = safety_router

# The primary brain: start by simulating "ACCELERATE"
primary_simulation = PredictiveJepaNode(action_to_simulate="ACCELERATE", next_node=safety_router)

# ==========================================
# 5. Execution (Setting the Robot on a Collision Course)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" LÁR JEPA WORLD MODEL DEMO")
    print("="*50)
    
    # We start dangerously close to the wall (X=8), going fast (V=5).
    # If the robot accelerates, it WILL hit the wall (X >= 10).
    initial_math_state = {
        "x_pos": 8, 
        "velocity": 5
    }
    
    executor = GraphExecutor()
    audit_log = list(executor.run_step_by_step(
        start_node=primary_simulation,
        initial_state=initial_math_state
    ))
    
    print("\n--- FINAL STATE ---")
    import pprint
    pprint.pprint(audit_log[-1]["state_diff"])
    print("✅ Notice how the Agent avoided the crash entirely by running an internal world model.")
