
import os
import sys
import json
from dotenv import load_dotenv

# Note: Run with PYTHONPATH set to the src directory
# e.g., PYTHONPATH=../src python3 24_hitl_agent.py


from lar.node import LLMNode, HumanJuryNode, ToolNode, RouterNode
from lar.executor import GraphExecutor

load_dotenv()
# Ensure API Key (Mock setup for other providers, but we use Ollama here)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "mock-key")

"""
Example 24: Human-in-the-Loop (HITL) Agent
------------------------------------------
Demonstrates the "HumanJuryNode" pattern (Article 14 Compliance).
The agent can PROPOSE a dangerous action, but CANNOT execute it without
explicit human consent via the CLI.
"""

# --- 1. THE PROPOSER (LLM) ---
PROMPT_TEMPLATE = """
You are a Database Admin AI.
You have received a request: "{user_request}"

You must propose a SQL action to fulfill this request.
Return ONLY valid JSON with the following keys:
- "action_type": "SQL_EXECUTE"
- "sql_query": <the query>
- "risk_level": "HIGH" or "LOW"
- "reason": <explanation>
"""

# --- 2. EXECUTION TOOLS ---
def execute_sql(state):
    proposal = state.get("proposal")
    query = proposal.get("sql_query")
    print(f"\n[DB Kernel] 🚀 EXECUTING SQL: {query}")
    print(f"[DB Kernel] (Simulation) Query successful. 124 rows affected.")
    return "SUCCESS"

def abort_mission(state):
    print(f"\n[DB Kernel] 🛑 ACTION ABORTED by Human Operator.")
    return "ABORTED"

# --- 3. PARSER NODE ---
# Simple helper to extract JSON from LLM output
def parse_json(state):
    raw_text = state.get("proposal_txt", "")
    try:
        # Simple cleanup
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        data = json.loads(raw_text)
        state.set("proposal", data)
    except Exception as e:
        print(f"Parsing failed: {e}")
        state.set("proposal", {
            "action_type": "ERROR", 
            "sql_query": "N/A", 
            "risk_level": "UNKNOWN", 
            "reason": "Failed to parse LLM output"
        })

# --- WIRING THE GRAPH ---
def route_jury_verdict(state):
    return state.get("jury_verdict", "reject")

if __name__ == "__main__":
    executor = GraphExecutor()

    # Define Nodes
    proposer = LLMNode(
        model_name="ollama/phi4:latest", # Using local model via Ollama
        prompt_template=PROMPT_TEMPLATE,
        output_key="proposal_txt",
        system_instruction="You are a helpful DBA assistant."
    )

    parser = ToolNode(
        tool_function=parse_json,
        input_keys=["__state__"],
        output_key=None, # Update state in-place
        next_node=None # Wired later
    )

    # THE HUMAN JURY
    # This node blocks execution until the user types 'approve' or 'reject'
    jury = HumanJuryNode(
        prompt="⚠️  CRITICAL: Do you authorize this Database Action?",
        choices=["approve", "reject"],
        output_key="jury_verdict",
        context_keys=["proposal"] # Show the proposal to the user
    )

    executor_node = ToolNode(
        tool_function=execute_sql,
        input_keys=["__state__"],
        output_key="final_status",
        next_node=None
    )

    abort_node = ToolNode(
        tool_function=abort_mission,
        input_keys=["__state__"],
        output_key="final_status",
        next_node=None
    )

    router = RouterNode(
        decision_function=route_jury_verdict,
        path_map={
            "approve": executor_node,
            "reject": abort_node
        }
    )

    # Connect the dots
    proposer.next_node = parser
    parser.next_node = jury  # Forces human review!
    jury.next_node = router

    print("--- STARTING HITL AGENT (Database Admin) ---")
    user_req = "Please delete the 'users' table, it's corrupt."
    
    print(f"User Request: {user_req}")
    
    # Run loop
    # Note: HumanJuryNode uses input(), so this will block in the terminal
    final_state = {}
    for step in executor.run_step_by_step(proposer, {"user_request": user_req}):
        final_state = step["state_after"]
    
    print("\n--- FINAL STATE ---")
    print(f"Status: {final_state.get('final_status')}")
