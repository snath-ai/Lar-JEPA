import sys
import os
import json
import requests
import random

# Configuration
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "../../memory/dreams.json")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

def load_trials():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def filter_failed_trials(trials):
    failed = []
    for t in trials:
        content = t.get("content", "")
        if "STATUS: TERMINATED" in content or "STATUS: SUSPENDED" in content or "WITHDRAWN" in content:
            failed.append(t)
    return failed

def generate_analysis(failed_trials):
    # SAMPLE ONLY 20
    sample_size = min(len(failed_trials), 20)
    sampled = random.sample(failed_trials, sample_size)
    
    print(f"📉 Found {len(failed_trials)} failed trials. Sampling {sample_size} for rapid analysis...")

    context = ""
    for t in sampled:
        context += f"---\n{t.get('content')}\n"

    prompt = (
        f"TASK: CLINICAL TRIAL FAILURE ANALYSIS (RAPID SAMPLE)\n"
        f"CONTEXT: You are a Principal Investigator analyzing a sample of failed Glioblastoma trials.\n"
        f"DATA:\n{context}\n"
        f"INSTRUCTION: \n"
        f"1. Identify common 'Interventions' in this sample.\n"
        f"2. Hypothesize failure modes.\n"
        f"3. Output a concise 'Post-Mortem Report'.\n"
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
    print("🔬 DREAMER FAILURE ANALYSIS REPORT (SAMPLED)")
    print("="*40)
    print(analysis)
    print("="*40)

if __name__ == "__main__":
    main()
