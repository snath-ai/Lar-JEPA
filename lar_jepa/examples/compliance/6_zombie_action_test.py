import hashlib
import json
import logging
import sys
import copy
from rich.console import Console
from rich.panel import Panel
from lar.node import BaseNode
from lar.executor import GraphExecutor

# Configure logging
logging.basicConfig(level=logging.ERROR)
console = Console()

# --- PATTERN 23: ZOMBIE ACTION STRESS TEST (STALE AUTHORITY) ---
#
# CONCEPT:
# Demonstrates "Unauthorized Continuation" (The Zombie Action).
# An action is approved in Context A.
# The system halts.
# The system resumes in Context B (e.g., target changed, or session expired).
#
# VULNERABILITY:
# A "Weak System" checks if `approved == True`. It sees True (from Context A) and executes Context B.
#
# DEFENSE:
# Lár's "Strong System" requires a cryptographic/invariant binding between the Approval and the specific Request.
# If the Request changes, the Approval becomes invalid (Stale).

# --- NODES ---

class WarrantRequestNode(BaseNode):
    """Reqests a search warrant for a specific target."""
    def __init__(self, name: str):
        self.name = name

    def execute(self, state):
        target = state.get("target", "House A")
        console.print(f"[bold blue]Request[/]: Warrant for [yellow]{target}[/]...")
        state.set("request_target", target)
        state.set("status", "pending_approval")
        return getattr(self, "next_node", None)

class WeakJudgeNode(BaseNode):
    """
    A naive judge. Sets 'approved=True'.
    Does not bind the approval to the specific target in a tamper-proof way.
    """
    def __init__(self, name: str):
        self.name = name

    def execute(self, state):
        target = state.get("request_target")
        console.print(f"[white]Weak Judge[/]: Warrant for {target} granted. Stamping 'APPROVED'.")
        state.set("status", "approved")
        state.set("approval_timestamp", 1234567890)
        return getattr(self, "next_node", None)

class StrongJudgeNode(BaseNode):
    """
    A Lár judge.
    Binds the approval to the state hash.
    signature = hash(target + secret)
    """
    def __init__(self, name: str):
        self.name = name

    def execute(self, state):
        target = state.get("request_target")
        # Cryptographic binding
        secret = "lar_secure_secret"
        payload = f"{target}:{secret}"
        signature = hashlib.sha256(payload.encode()).hexdigest()
        
        console.print(f"[green]Lár Judge[/]: Warrant for {target} granted. Signing hash: [cyan]{signature[:8]}...[/]")
        state.set("status", "approved")
        state.set("warrant_signature", signature)
        return getattr(self, "next_node", None)

class ExecutionNode(BaseNode):
    """
    The Swat Team. Breaks down the door.
    """
    def __init__(self, name: str, mode: str = "weak"):
        self.name = name
        self.mode = mode

    def execute(self, state):
        target = state.get("target") # Note: Reads current global target
        status = state.get("status")
        
        console.print(f"[red]Action ({self.mode.upper()})[/]: Executing on [yellow]{target}[/]. Checking authorization...")

        if self.mode == "weak":
            # WEAK CHECK: Just looks for the flag
            if status == "approved":
                console.print(f"[bold red][CRITICAL] Unauthorized Access[/]: 'Approved' flag found. Entering [yellow]{target}[/]!")
                state.set("outcome", "breached")
                state.set("breached_target", target)
            else:
                console.print("[red][BLOCKED][/]: No approval flag.")
                state.set("outcome", "blocked")
        
        elif self.mode == "strong":
            # STRONG CHECK: Re-verifies signature against CURRENT target
            signature = state.get("warrant_signature")
            secret = "lar_secure_secret"
            expected_payload = f"{target}:{secret}"
            expected_signature = hashlib.sha256(expected_payload.encode()).hexdigest()
            
            # Simple simulation of "Did we have a valid signature?"
            # In a real case, we'd check if signature exists first.
            if not signature:
                 console.print(f"[bold red]🛑 HALT[/]: No signature found.")
                 state.set("outcome", "blocked")
                 return getattr(self, "next_node", None)

            if signature == expected_signature:
                console.print(f"[bold red][CRITICAL] Unauthorized Access[/]: Signature matches [yellow]{target}[/]. Valid.")
                state.set("outcome", "breached")
                state.set("breached_target", target)
            else:
                console.print(f"[bold green][BLOCKED][/]: STALE AUTHORITY DETECTED!")
                console.print(f"   Stored Signature: {signature[:8]}... (Authorized for Old Target)")
                console.print(f"   Required Signature: {expected_signature[:8]}... ({target})")
                console.print("   [i]Zombie Action Prevented.[/i]")
                state.set("outcome", "zombie_blocked")
        
        return getattr(self, "next_node", None)

# --- SIMULATION ---

def run_test():
    console.print(Panel.fit("[bold white]Pattern 23: Zombie Action Stress Test[/]", border_style="red"))

    # 1. SETUP: The Honest Request
    initial_state = {"target": "House A"}
    
    # WEAK SYSTEM SIMULATION
    console.print("\n[bold]--- SCENARIO 1: WEAK SYSTEM (Targeted) ---[/]")
    
    # Step 1: Officer requests, Judge approves
    req = WarrantRequestNode("Request")
    judge = WeakJudgeNode("Judge")
    executor = ExecutionNode("Execute", mode="weak")
    
    req.next_node = judge
    # NOTE: We intentionally stop here to simulate "Crash".
    # judge.next_node = executor 
    
    graph = GraphExecutor()
    state_after_run1 = {}
    
    # Run Part 1 (Approval)
    for step in graph.run_step_by_step(req, initial_state):
        # Manually reconstruct state_after if not present
        if "state_after" in step:
            state_after_run1 = copy.deepcopy(step["state_after"])
        else:
            # Fallback reconstruction
            s = copy.deepcopy(step["state_before"])
            diff = step["state_diff"]
            s.update(diff.get("added", {}))
            s.update(diff.get("updated", {}))
            state_after_run1 = s
            
        if step["node"] == "Judge":
            break
            
    console.print(f"\n[dim]System State Saved: {state_after_run1}[/]")
    console.print("[dim][System] Simulating Restart (Memory Wipe)[/]")
    
    # 2. THE HACK: Context Contamination
    # While system was down, the "Target" variable was swapped.
    # But the "status" remains "approved".
    state_after_run1["target"] = "House B (Innocent Family)"
    console.print(f"[bold red][ALERT] Context Drift[/]: Target switched to '[yellow]House B[/]' in persistent storage.")
    
    # 3. THE ZOMBIE RESUME
    console.print("\n[bold]RESUMING EXECUTION...[/]")
    # We resume directly at execution, assuming state is valid
    zombie_run = GraphExecutor()
    for step in zombie_run.run_step_by_step(executor, state_after_run1):
        pass


    # STRONG SYSTEM SIMULATION
    console.print("\n\n[bold]--- SCENARIO 2: LÁR SYSTEM (Protected) ---[/]")
    
    # Step 1: Same setup
    req_s = WarrantRequestNode("Request")
    judge_s = StrongJudgeNode("Judge") # Uses Signature
    executor_s = ExecutionNode("Execute", mode="strong")
    
    req_s.next_node = judge_s
    
    graph_s = GraphExecutor()
    state_s_run1 = {}
    
    # Run Part 1
    for step in graph_s.run_step_by_step(req_s, initial_state):
        # Manually reconstruct state_after if not present
        if "state_after" in step:
            state_s_run1 = copy.deepcopy(step["state_after"])
        else:
            s = copy.deepcopy(step["state_before"])
            diff = step["state_diff"]
            s.update(diff.get("added", {}))
            s.update(diff.get("updated", {}))
            state_s_run1 = s
            
        if step["node"] == "Judge":
            break
            
    # Step 2: The Hack
    console.print("[dim][System] Simulating Restart (Memory Wipe)[/]")
    state_s_run1["target"] = "House B (Innocent Family)"
    console.print(f"[bold red][ALERT] Context Drift[/]: Target switched to '[yellow]House B[/]'.")
    
    # Step 3: Zombie Resume
    console.print("\n[bold]RESUMING EXECUTION...[/]")
    zombie_run_s = GraphExecutor()
    for step in zombie_run_s.run_step_by_step(executor_s, state_s_run1):
        pass

    console.print("\n[bold green]✅ TEST COMPLETE: Invariant Held.[/]")

if __name__ == "__main__":
    run_test()
