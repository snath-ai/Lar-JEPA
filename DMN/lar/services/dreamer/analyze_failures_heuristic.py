import sys
import os
import json
import re
from collections import Counter

# Configuration
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "../../memory/dreams.json")

def load_trials():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def filter_failed_trials(trials):
    failed = []
    for t in trials:
        content = t.get("content", "")
        # Check for failure keywords in the narrative constructed by ingest_trials.py
        if "STATUS: TERMINATED" in content or "STATUS: SUSPENDED" in content or "WITHDRAWN" in content:
            failed.append(t)
    return failed

def extract_interventions(trial_content):
    # Narrative format: "Study [Title] tested [Intervention] on..."
    # We'll use a simple regex to find the intervention part if possible, 
    # or just look for common drug names/types.
    
    # Heuristic: split by "tested " and " on "
    try:
        part1 = trial_content.split("tested ")[1]
        intervention = part1.split(" on ")[0]
        return intervention
    except IndexError:
        return None

def analyze_failures_heuristic(failed_trials):
    print(f"📉 Analyzing {len(failed_trials)} failed trials using Heuristic NLP (Zero-Compute)...")
    
    interventions = []
    failure_reasons = []

    for t in failed_trials:
        content = t.get("content", "")
        
        # Extract Intervention
        inv = extract_interventions(content)
        if inv:
            # Clean up: remove "DRUG: " prefix if present, split multi-agent
            invs = inv.replace("DRUG: ", "").replace("RADIATION: ", "").split("|")
            interventions.extend([i.strip() for i in invs])
        
        # Heuristic for Reason (often in Summary)
        if "toxicity" in content.lower():
            failure_reasons.append("Toxicity")
        if "efficacy" in content.lower():
            failure_reasons.append("Lack of Efficacy")
        if "accrual" in content.lower():
            failure_reasons.append("Low Accrual")
            
    # Count frequencies
    inv_counts = Counter(interventions).most_common(10)
    reason_counts = Counter(failure_reasons).most_common(5)
    
    return inv_counts, reason_counts

def main():
    trials = load_trials()
    failed = filter_failed_trials(trials)
    
    if not failed:
        print("No failed trials found.")
        return

    inv_stats, reason_stats = analyze_failures_heuristic(failed)
    
    print("\n" + "="*50)
    print("🔬 HEURISTIC FAILURE ANALYSIS REPORT (CPU SAFE)")
    print("="*50)
    print(f"Total Failed/Terminated Trials: {len(failed)}")
    print("\n[TOP 10 FAILED INTERVENTIONS]")
    for drug, count in inv_stats:
        print(f"  • {drug}: {count} failures")
        
    print("\n[DETECTED FAILURE SIGNALS]")
    for reason, count in reason_stats:
        print(f"  • {reason}: {count} mentions")
    print("="*50)

if __name__ == "__main__":
    main()
