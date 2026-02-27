# tests/test_core.py

import pytest
# Removed PrintStateNode import
from lar import GraphExecutor, AddValueNode, GraphState, apply_diff 

def test_simple_graph_execution():
    """
    Tests a simple, linear A -> B graph.
    - Node A (AddValueNode) adds a value.
    - Node B (AddValueNode) marks the end.
    Asserts that both values are in the final state.
    """
    
    # 1. Arrange: Set up the graph and executor
    initial_state = {"user": "Solo Developer"}
    
    # Define the nodes in reverse order, from end to start
    
    # Node B: The end of the graph (replaces PrintStateNode)
    # This node will run, add a final marker, then return None, ending the graph.
    end_node = AddValueNode(
        key="test_status",
        value="COMPLETED",
        next_node=None # CRITICAL: Signals the end of execution
    )
    
    # Node A: The start of the graph
    start_node = AddValueNode(
        key="message", 
        value="Hello Lár!", 
        next_node=end_node
    )
    
    executor = GraphExecutor()

    # 2. Act: Run the graph and capture the full audit log
    audit_log = list(executor.run_step_by_step(
        start_node=start_node, 
        initial_state=initial_state
    ))
    
    # Reconstruct the final state from the log
    final_state_data = initial_state
    for step in audit_log:
        final_state_data = apply_diff(final_state_data, step["state_diff"])

    # 3. Assert: Check if the graph did its job
    
    # The audit log must contain exactly two steps
    assert len(audit_log) == 2 
    
    # Check that the final state contains the values from both nodes
    assert final_state_data is not None
    assert final_state_data.get("user") == "Solo Developer"
    assert final_state_data.get("message") == "Hello Lár!"
    assert final_state_data.get("test_status") == "COMPLETED"


def test_batch_node_budget_merge():
    from lar import BatchNode, BaseNode
    
    class DummySpendNode(BaseNode):
        def __init__(self, spend_amount, output_key, next_node=None):
            self.next_node = next_node
            self.spend_amount = spend_amount
            self.output_key = output_key
            
        def execute(self, state):
            current_budget = state.get("token_budget")
            if current_budget is not None:
                state.set("token_budget", current_budget - self.spend_amount)
            state.set(self.output_key, f"spent {self.spend_amount}")
            return self.next_node

    # Base budget is 1000
    initial_state = {"token_budget": 1000}
    
    # Node 1 spends 100, Node 2 spends 250
    node1 = DummySpendNode(spend_amount=100, output_key="result_1")
    node2 = DummySpendNode(spend_amount=250, output_key="result_2")
    
    end_node = AddValueNode(key="end", value=True, next_node=None)
    
    batch_node = BatchNode(
        nodes=[node1, node2],
        next_node=end_node
    )
    
    executor = GraphExecutor()
    log = list(executor.run_step_by_step(start_node=batch_node, initial_state=initial_state))
    
    final_state = initial_state
    for step in log:
        final_state = apply_diff(final_state, step["state_diff"])
        
    assert final_state.get("result_1") == "spent 100"
    assert final_state.get("result_2") == "spent 250"
    
    # Total spend should be 350. Remaining budget: 1000 - 350 = 650.
    assert final_state.get("token_budget") == 650