"""
Example: Using Liquid Thinking (System 2) with Lár

Liquid Thinking is a powerful reasoning model that often outperforms larger models on logic tasks.
It is available via Ollama: `ollama pull liquid-thinking`

This example demonstrates how Lár automatically captures its reasoning trace.
"""

import sys
from lar import GraphState, LLMNode, GraphExecutor

# --- Configuration ---
# Ensure you have run: ollama pull liquid-thinking
MODEL_NAME = "ollama/liquid-thinking:latest"

print(f"🚀 Lár Liquid Thinking Example (using {MODEL_NAME})")
print("=" * 60)

# 1. Define the Node
liquid_node = LLMNode(
    model_name=MODEL_NAME,
    prompt_template="Solve this logic puzzle: {input}",
    output_key="answer",
    # CRITICAL: Force the model to use the tags we want to capture
    # Without this, local models like liquid-thinking might output raw reasoning mixed with the answer.
    # By enforcing the XML structure, Lár can cleanly separate the two.
    system_instruction="You are a reasoning engine. You MUST enclose your internal monologue in <think> tags. After the tags, provide the final answer."
)

# 2. Define State
initial_state = {
    # Liquid models excel at lateral thinking puzzles
    "input": "A man pushes his car to a hotel and tells the owner he's bankrupt. Why? Explain your reasoning step-by-step."
}

# 3. Execution
executor = GraphExecutor(log_dir="reasoning_logs")

print(f"\n[Puzzle]: {initial_state['input']}\n")
print(f"Thinking... (Model: {MODEL_NAME})")

# Run the graph
try:
    steps = list(executor.run_step_by_step(liquid_node, initial_state))
except Exception as e:
    print(f"❌ Execution failed: {e}")
    sys.exit(1)

if not steps:
    print("❌ No steps executed.")
    sys.exit(1)

# 4. Inspect Results
final_step = steps[-1]
final_state = final_step["state_after"]
metadata = final_step["run_metadata"]

# Retrieve the captured reasoning trace
reasoning_trace = metadata.get("reasoning_content")
final_answer = final_state.get("answer")

print("-" * 60)
if reasoning_trace:
    print(f"🧠 [Reasoning Trace Captured] ({len(reasoning_trace)} chars):")
    print(f"{reasoning_trace[:500]}...\n[... truncated ...]")
else:
    print("⚠️ [No Reasoning Trace] The model didn't output distinct reasoning metadata.")
    print("   (Check if your local model supports metadata or output <think> tags)")

print("-" * 60)
print(f"✅ [Final Answer] (Clean):")
print(final_answer)
print("-" * 60)

print(f"\n📂 Full log saved to: reasoning_logs/")
