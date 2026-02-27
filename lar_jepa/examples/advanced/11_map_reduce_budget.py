"""
LÁR MAP-REDUCE & ECONOMIC CONSTRAINTS DEMO (v1.6.0)

This example demonstrates two new defensive mechanisms for production agents:
1.  **Memory Compression (Map-Reduce):** 
    Using `BatchNode` (Map) to fetch 3 separate detailed reports in parallel,
    followed immediately by a `ReduceNode` (Reduce) to summarize them and 
    explicitly DELETE the bloated raw context from the GraphState.
2.  **Economic Constraints (Token Budgets):** 
    Giving the agent a strict budget. We will see the agent successfully
    run the Map-Reduce, but then hit the budget ceiling on the final step.

Model: ollama/phi4:latest
"""

import sys
from pathlib import Path

# Add src to path for easy execution from inside 'examples/'
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lar import GraphState, GraphExecutor, LLMNode, BatchNode, ReduceNode, FunctionalNode
from dotenv import load_dotenv

load_dotenv()

def main():
    print("\n" + "="*60)
    print("LAR DEFENSIVE CONSTRAINTS & MAP-REDUCE PATTERN")
    print("="*60)
    print("This graph will execute 3 parallel 'research' tasks, merge them,")
    print("compress the memory to prevent context bloat, and then deliberately")
    print("exhaust a strict Token Budget to demonstrate economic constraints.")
    print("-" * 60 + "\n")

    # --- 1. The Map Phase (Parallel Workers) ---
    # We simulate reading three heavy documents
    researcher_1 = LLMNode(
        model_name="ollama/phi4:latest",
        prompt_template="Write a detailed 3-paragraph report on the history of AI in healthcare.",
        output_key="healthcare_report"
    )
    
    researcher_2 = LLMNode(
        model_name="ollama/phi4:latest",
        prompt_template="Write a detailed 3-paragraph report on the history of AI in finance.",
        output_key="finance_report"
    )

    researcher_3 = LLMNode(
        model_name="ollama/phi4:latest",
        prompt_template="Write a detailed 3-paragraph report on the history of AI in robotics.",
        output_key="robotics_report"
    )

    # The Reduce Phase: Read the 3 heavy reports, extract insights, and delete the raw text
    # The next node points to an LLMNode that will try to run but fail due to the budget
    final_attempt_node = LLMNode(
        model_name="ollama/phi4:latest",
        prompt_template="Now, write an extremely detailed book about these insights: {executive_summary}",
        output_key="final_book"
    )

    reducer = ReduceNode(
        model_name="ollama/phi4:latest",
        prompt_template="Summarize the core themes across these three sectors based ONLY on these reports:\\n1. Healthcare: {healthcare_report}\\n2. Finance: {finance_report}\\n3. Robotics: {robotics_report}\\n\\nProvide a 3-bullet executive summary.",
        input_keys=["healthcare_report", "finance_report", "robotics_report"],
        output_key="executive_summary",
        next_node=final_attempt_node
    )

    # Batch them together
    map_phase = BatchNode(
        nodes=[researcher_1, researcher_2, researcher_3],
        next_node=reducer
    )

    # --- 2. Initialize Executor & State ---
    executor = GraphExecutor(log_dir="lar_logs")
    
    # We give it a generous but constrained budget. The Map-Reduce phase should cost
    # around 1500-2000 tokens total. We will set the budget to 2500 so it can finish
    # the summarization but fail on the "final_book" attempt.
    initial_state = {"token_budget": 2500}

    print(f"[i] Initial Token Budget set to: {initial_state['token_budget']}")

    # --- 3. Execute Loop ---
    final_state = {}
    for step_log in executor.run_step_by_step(map_phase, initial_state):
        if step_log["outcome"] == "error":
            print(f"\\n[!] ERROR ENCOUNTERED: {step_log.get('error')}")
            break
        final_state = step_log.get("state_after", {})

    print("\n" + "="*50)
    print("EXECUTION TERMINATED")
    print("="*50)

    print("\n[VERIFICATION OF MEMORY COMPRESSION]")
    keys = final_state.keys()
    print(f"Does 'healthcare_report' exist? {'Yes' if 'healthcare_report' in keys else 'No (Deleted)'}")
    print(f"Does 'finance_report' exist? {'Yes' if 'finance_report' in keys else 'No (Deleted)'}")
    print(f"Does 'robotics_report' exist? {'Yes' if 'robotics_report' in keys else 'No (Deleted)'}")
    print(f"Does 'executive_summary' exist? {'Yes' if 'executive_summary' in keys else 'No'}")
    
    if "executive_summary" in final_state:
        print(f"\n[EXECUTIVE SUMMARY]:\n{final_state['executive_summary']}")
        
    print(f"\n[FINAL TOKEN BUDGET]: {final_state.get('token_budget')} tokens remaining.")

if __name__ == "__main__":
    main()
