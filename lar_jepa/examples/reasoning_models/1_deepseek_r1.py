"""
Example: Using Reasoning Models (System 2) with Lár

Lár v1.4.1+ automatically supports "Reasoning Trace" capture for models like:
- DeepSeek R1
- OpenAI o1
- Liquid Thinking (ollama/liquid-thinking)

How it works:
1. The model generates a "Thinking Process" (e.g., inside <think> tags or via API metadata).
2. Lár extracts this trace and stores it in `state["__last_run_metadata"]["reasoning_content"]`.
3. The main `state["output_key"]` contains ONLY the clean final answer.

This script demonstrates this using an Ollama model (DeepSeek R1 or Liquid Thinking).
"""

import sys
from lar import GraphState, LLMNode, GraphExecutor

# --- Configuration ---
# You can swap this with "ollama/deepseek-r1:7b", "openai/o1", etc.
MODEL_NAME = "ollama/liquid-thinking:latest" 

print(f"🚀 Lár Reasoning Model Example (using {MODEL_NAME})")
print("=" * 60)

# 1. Define the Node
# We use a standard LLMNode. No special configuration needed!
reasoning_node = LLMNode(
    model_name=MODEL_NAME,
    prompt_template="Solve this logic puzzle: {input}",
    output_key="answer",
    # Optional: Encourage usage of <think> tags if the model is stubborn
    system_instruction="You are a reasoning model. Always output your internal monologue inside <think> tags before your final answer."
)

# 2. Define State
initial_state = {
    "input": "If I have 3 apples and eat 1, and then buy 2 more, how many do I currently have? Explain step-by-step."
}

# 3. Execution
executor = GraphExecutor(log_dir="reasoning_logs")

print(f"\n[Input]: {initial_state['input']}\n")
print(f"Thinking... (Model: {MODEL_NAME})")

# Run the graph
steps = list(executor.run_step_by_step(reasoning_node, initial_state))

if not steps:
    print("❌ Execution failed (no steps returned).")
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
    # Print the first 500 chars to avoid cluttering the terminal
    print(f"{reasoning_trace[:500]}...\n[... truncated ...]")
else:
    print("⚠️ [No Reasoning Trace] The model didn't output distinct reasoning metadata or <think> tags.")
    print("   (Check if your local model is quantized/configured to strip reasoning)")

print("-" * 60)
print(f"✅ [Final Answer] (Clean):")
print(final_answer)
print("-" * 60)

print(f"\n📂 Full log saved to: reasoning_logs/")
print("   Inspect the JSON log to see the FULL reasoning trace under 'run_metadata'.")
