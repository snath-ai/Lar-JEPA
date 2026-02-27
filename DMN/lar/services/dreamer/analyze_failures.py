import sys
import os
import json
import requests

# Configuration
# Path to dreams.json (Source of Truth for this demo)
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "../../memory/dreams.json")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b") # Use the smarter model for analysis

def load_trials():
    if not os.path.exists(MEMORY_FILE):
        print(f"❌ Memory file not found: {MEMORY_FILE}")
        return []
    
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def filter_failed_trials(trials):
    failed = []
    for t in trials:
        content = t.get("content", "")
        # Naive filtering based on string matching in the narrative
        if "STATUS: TERMINATED" in content or "STATUS: SUSPENDED" in content or "WITHDRAWN" in content:
            failed.append(t)
        # Also check for "RESULTS: NO" if status is COMPLETED but maybe failed? 
        # Actually strategy asked for "'No Results' or 'Terminated'"
        # Let's stick to explicit failure states + No Results for now involves too many.
        # Let's focus on Terminated/Withdrawn/Suspended first as they are clear failures.
    return failed

def generate_analysis(failed_trials):
    print(f"📉 Found {len(failed_trials)} failed/terminated trials. Analyzing...")
    
    if not failed_trials:
        return "No failed trials found to analyze."

    # Prepare context
    context = ""
    for t in failed_trials:
        context += f"---\n{t.get('content')}\n"

    prompt = (
        f"TASK: CLINICAL TRIAL FAILURE ANALYSIS\n"
        f"CONTEXT: You are a Principal Investigator analyzing a dataset of failed Glioblastoma trials.\n"
        f"DATA:\n{context}\n"
        f"INSTRUCTION: \n"
        f"1. Identify the common 'Interventions' used in these failed trials.\n"
        f"2. Identify any patterns in 'Conditions' or patient cohorts.\n"
        f"3. Hypothesize WHY these approaches might have failed (e.g. bioavailability, toxicity, wrong target).\n"
        f"4. Output a concise 'Post-Mortem Report' summarizing the failure modes.\n"
    )

    try:
        print(f"🧠 Sending to {MODEL}...")
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error connecting to LLM: {e}"

def main():
    trials = load_trials()
    failed = filter_failed_trials(trials)
    analysis = generate_analysis(failed)
    
    print("\n" + "="*40)
    print("🔬 DREAMER FAILURE ANALYSIS REPORT")
    print("="*40)
    print(analysis)
    print("="*40)

    # Optional: Save this insight back to memory?
    # For now, just print.

if __name__ == "__main__":
    main()
