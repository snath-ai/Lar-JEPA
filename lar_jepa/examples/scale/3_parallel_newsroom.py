# For setup instructions, see: https://docs.snath.ai/guides/litellm_setup/
"""
Example 15: The Newsroom (True Parallel Execution)
--------------------------------------------------
Demonstrates the "Fan-Out / Fan-In" pattern using BatchNode for *true* parallelism.

Scenario: A Newsroom covering a breaking topic.
1. Batch Fan-Out: Fact, Opinion, and History reporters run AT THE SAME TIME.
2. Fan-In: Editor merges the results.

Model: ollama/phi4:latest
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from lar.node import LLMNode, ToolNode, BatchNode
from lar.executor import GraphExecutor

# ============================================================================
# 1. The Reporters (Fan-Out)
# ============================================================================

# Reporter A: The Fact Checker
fact_reporter = LLMNode(
    model_name="ollama/phi4:latest",
    prompt_template="""
    Topic: {topic}
    List 3 key factual bullet points about this topic.
    Focus on dates, numbers, and verifiable events.
    """,
    output_key="facts_section",
    system_instruction="You are a strict fact-checker."
)

# Reporter B: The Opinion Columnist
opinion_columnist = LLMNode(
    model_name="ollama/phi4:latest",
    prompt_template="""
    Topic: {topic}
    Write a short, provocative 2-sentence opinion on this.
    """,
    output_key="opinion_section",
    system_instruction="You are a bold opinion writer."
)

# Reporter C: The Historian
historian = LLMNode(
    model_name="ollama/phi4:latest",
    prompt_template="""
    Topic: {topic}
    Provide 1 sentence of historical context for this.
    """,
    output_key="history_section",
    system_instruction="You are a historian."
)

# ============================================================================
# 2. The Editor (Fan-In)
# ============================================================================

editor_in_chief = LLMNode(
    model_name="ollama/phi4:latest",
    prompt_template="""
    Create a final Newsletter Issue about '{topic}'.
    
    Use these raw reports:
    
    [FACTS]
    {facts_section}
    
    [OPINION]
    {opinion_section}
    
    [HISTORY]
    {history_section}
    
    Format nicely with emojis and headers.
    """,
    output_key="final_newsletter",
    system_instruction="You are a prestigious Editor-in-Chief."
)

# ============================================================================
# Wiring
# ============================================================================

# TRUE PARALLELISM:
# Instead of A->B->C, we wrap them in a BatchNode.

news_desk = BatchNode(
    nodes=[fact_reporter, opinion_columnist, historian],
    next_node=editor_in_chief
)

# Editor stops (Next=None)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    executor = GraphExecutor()
    topic = "The Future of Space Exploration"
    
    print(f"\n📰 EXTRA! EXTRA! Covering: '{topic}'")
    print("⚡ Spinning up the News Desk (Parallel Mode)...")
    
    start_time = time.time()
    
    initial_state = {"topic": topic}
    
    # Start at the BatchNode
    for step in executor.run_step_by_step(news_desk, initial_state):
        node = step['node']
        
        # Simple progress indicator
        if "BatchNode" in node:
             print(f"  >>> Batch Processing in progress...")
        elif "LLMNode" in str(type(step.get('node_object'))):
            print(f"  ... {node} is writing ...")
            
        # Check for final output in usage diff
        state_diff = step.get('state_diff', {})
        if "final_newsletter" in state_diff:
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n" + "="*50)
            print(f"📬 FINAL NEWSLETTER (Generated in {duration:.2f}s)")
            print("="*50)
            print(state_diff['final_newsletter'])
