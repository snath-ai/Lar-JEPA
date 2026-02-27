# For setup instructions, see: https://docs.snath.ai/guides/litellm_setup/
import os
import sys
import json
from lar import GraphState, GraphExecutor, LLMNode, BatchNode

# ------------------------------------------------------------------------------
# Example 17: The A/B Tester (Agentic Ops)
# ------------------------------------------------------------------------------
# Use Case: Evaluating prompt variations or different models in parallel.
# Pattern:  Fan-Out (BatchNode) -> Fan-In (Judge Node).
# ------------------------------------------------------------------------------

# 1. Define the Judge (The Fan-In)
# Must be defined first so it can be passed as 'next_node' to the BatchNode
judge_node = LLMNode(
    model_name="ollama/phi4",
    prompt_template="""
    You are an expert evaluator. Compare these two responses to the user's query and pick the best one.
    
    User Query: {customer_query}
    
    Response A (Formal): {response_a}
    
    Response B (Pirate): {response_b}
    
    Which is better? Reply with just 'A' or 'B' and a brief reason.
    """,
    output_key="judgement",
    next_node=None # End of graph
)

# 2. Define Prompt Variation A (Helpful)
model_a = LLMNode(
    model_name="ollama/phi4",
    prompt_template="You are a helpful customer service agent. Query: {customer_query}",
    output_key="response_a",
    next_node=None # BatchNode ignores individual next_nodes
)

# 3. Define Prompt Variation B (Sarcastic Pirate)
model_b = LLMNode(
    model_name="ollama/phi4",
    prompt_template="You are a sarcastic pirate. Answer this query: {customer_query}",
    output_key="response_b",
    next_node=None # BatchNode ignores individual next_nodes
)

# 4. Define the Batch Runner
# Runs A and B in parallel, then moves to Judge
batch_runner = BatchNode(
    nodes=[model_a, model_b],
    next_node=judge_node
)

def run_ab_test():
    # Setup state
    initial_state = {
        "customer_query": "My internet is down and I am angry."
    }
    
    # Run the experiment
    print(f"\n🧪 STARTING A/B TEST: '{initial_state['customer_query']}'")
    
    executor = GraphExecutor() # Logic is in run_step_by_step
    
    final_step = None
    for step in executor.run_step_by_step(batch_runner, initial_state):
        print(f"Step {step['step']}: {step['node']} -> {step['outcome']}")
        final_step = step
    
    # Extract results from the final state diffs (or reconstruct state if needed)
    # The 'Judge' node runs last, so its output 'judgement' is in the final diff.
    # The 'BatchNode' runs before, so 'response_a' and 'response_b' are in history.
    
    # To display nicely, we need to know what 'response_a' and 'response_b' were.
    # We can inspect the 'state_before' of the final step (which is the state passed TO the judge)
    state_for_judge = final_step['state_before'] # This has responses but not judgement
    judgement = final_step['state_diff'].get('judgement') # The result of judge
    
    print("\n" + "="*50)
    print("📢 RESULTS")
    print("="*50)
    print(f"\n🅰️  FORMAL:\n{state_for_judge.get('response_a')}")
    print(f"\n🏴‍☠️  PIRATE:\n{state_for_judge.get('response_b')}")
    print("\n" + "-"*50)
    print(f"\n⚖️  JUDGEMENT:\n{judgement}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_ab_test()
