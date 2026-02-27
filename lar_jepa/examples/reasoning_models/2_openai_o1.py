"""
Example: Using OpenAI o1 (Reasoning Model) with Lár

OpenAI's o1 series (o1-preview, o1-mini) are reasoning models.
Lár + LiteLLM handles the API differences automatically (e.g., converting 'max_tokens' to 'max_completion_tokens').

Reasoning traces for o1 are strictly controlled by OpenAI and may not always be visible in the API response 
depending on your tier/policy, but if they are returned, Lár captures them.
"""

from lar import GraphState, LLMNode, GraphExecutor
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Ensure OPENAI_API_KEY is set in your .env
MODEL_NAME = "gpt-4o" # or "o1-mini" / "o1-preview" when you have access

print(f"🚀 Lár OpenAI o1 Example (using {MODEL_NAME})")
print("=" * 60)

# 1. Define the Node
# Note: o1 models often don't support 'system' role. 
# LiteLLM/LAr handles this, but it's best to put instructions in the user prompt.
o1_node = LLMNode(
    model_name=MODEL_NAME,
    prompt_template="Solve this complex problem: {input}",
    output_key="answer",
    # o1 supports explicit reasoning effort configuration in some versions
    generation_config={
        "max_completion_tokens": 1000 
    }
)

# 2. Define State
initial_state = {
    "input": "Write a Python script that outputs its own source code (Quine) without reading text files."
}

# 3. Executiom
executor = GraphExecutor(log_dir="reasoning_logs")

print(f"Sending to {MODEL_NAME}...")
# executor.run_step_by_step(...) 
# (This is just a template, we won't run it without an API key)

if __name__ == "__main__":
    print("\n[Setup Complete]")
    print(f"To run this, ensure OPENAI_API_KEY is set and execute:")
    print(f"  python examples/reasoning_models/2_openai_o1.py")
    
    # Check for key
    if os.getenv("OPENAI_API_KEY"):
        steps = list(executor.run_step_by_step(o1_node, initial_state))
        final = steps[-1]["state_after"]
        print(f"\n✅ Answer:\n{final['answer'][:200]}...")
    else:
        print("\n⚠️ Skipped execution: OPENAI_API_KEY not found.")
