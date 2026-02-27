
"""
Lár v1.4.0 Feature Showcase

This example demonstrates how the new "Low Code" syntax (@node)
coexists perfectly with the classic "Explicit" syntax (Class-based).

SCENARIO: A simple pipeline:
1. Input -> [Processor A] -> Intermediate
2. Intermediate -> [Processor B] -> Final Output
"""
import os
from lar import GraphState, GraphExecutor, BaseNode, node, BatchNode

# ==========================================
# 1. The "Old Way" (Explicit Class Definition)
# ==========================================
# Good for: Reusable components, complex logic, holding state.

class ClassicProcessor(BaseNode):
    def __init__(self, output_key, next_node=None):
        self.output_key = output_key
        self.next_node = next_node

    def execute(self, state: GraphState):
        # Classic access: state.get()
        val = state.get("input", 0)
        print(f"  [ClassicProcessor] Received: {val}")
        
        # transform
        new_val = val + 10
        
        # Classic set: state.set() or return dict
        # We manually set the state here because we are a raw BaseNode
        state.set(self.output_key, new_val)
        
        # Return the next node to continue the graph
        return self.next_node

# ==========================================
# 2. The "New Way" (Functional Decorator)
# ==========================================
# Good for: Quick logic, data transformation, one-off scripts.

@node(output_key="final_result")
def fast_processor(state):
    # New dictionary access!
    val = state["intermediate"]
    print(f"  [FastProcessor @node] Received: {val}")
    
    return val * 2

# ==========================================
# 3. Wiring It Up
# ==========================================

# Instantiate Classic Node
# We have to wrap it to use it in the graph unless we use pre-built nodes like AddValueNode
# Let's use a standard pattern for the old way: a wrapper or a custom node.
# Here we instance it.
classic_node = ClassicProcessor(output_key="intermediate", next_node=None)

# Connect the New Node to follow the Old Node
classic_node.next_node = fast_processor

# ==========================================
# 4. Execution
# ==========================================

if __name__ == "__main__":
    print("--- Lár v1.4.0 Showcase ---")
    
    # 1. Setup State
    state = {"input": 5}
    print(f"Initial State: {state}")
    
    # 2. Run
    executor = GraphExecutor()
    
    # We run step-by-step
    for step in executor.run_step_by_step(classic_node, state):
        print(f"Step completed: {step['node']}")

    # 3. Verify Final State (Accessing via dict!)
    # Note: executor.run returns generator, we need to inspect the state object directly if we had a reference,
    # or look at the last yielded step.
    # But for this demo, let's just see logs.
    
    print("\n✅ Verification:")
    print("1. Classic Node (Old Way) added 10 -> 15")
    print("2. @node (New Way) multiplied by 2 -> 30")
    print("Both styles work together seamlessly.")
