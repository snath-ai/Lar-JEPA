import sys
import os
import csv
import json
import requests
import time
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from brain.hippocampus import Hippocampus

# Configuration
TRIALS_CSV = os.path.join(os.path.dirname(__file__), "../../trials/ctg-studies.csv")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

def get_embedding(text):
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": DEFAULT_MODEL, "prompt": text}
        )
        if response.status_code == 200:
            return response.json().get("embedding", [])
        else:
            print(f"⚠️ Embed failed: {response.text}")
            return []
    except Exception as e:
        print(f"⚠️ Embed Error: {e}")
        return []

def row_to_narrative(row):
    """
    Converts a CSV row into a readable narrative for the memory.
    """
    title = row.get("Study Title", "Unknown Study")
    condition = row.get("Conditions", "Unknown Condition")
    intervention = row.get("Interventions", "Unknown Intervention")
    summary = row.get("Brief Summary", "No summary provided.")
    status = row.get("Study Status", "Unknown Status")
    results = row.get("Study Results", "No results reported")

    narrative = (
        f"CLINICAL TRIAL: {title}\n"
        f"CONDITION: {condition}\n"
        f"INTERVENTION: {intervention}\n"
        f"STATUS: {status}\n"
        f"SUMMARY: {summary}\n"
        f"RESULTS: {results}"
    )
    return narrative

def main():
    print(f"🚀 Starting Clinical Trials Ingestion from {TRIALS_CSV}...")
    
    if not os.path.exists(TRIALS_CSV):
        print(f"❌ File not found: {TRIALS_CSV}")
        return

    # Initialize Hippocampus
    hippocampus = Hippocampus()

    count = 0
    with open(TRIALS_CSV, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            narrative = row_to_narrative(row)
            
            # Generate Embedding
            embedding = get_embedding(narrative)
            
            if not embedding:
                print(f"⚠️ Skipping {row.get('NCT Number')} due to embedding failure.")
                continue

            # Metadata
            metadata = {
                "source": "clinical_trials_gov",
                "nct_id": row.get("NCT Number", ""),
                "type": "clinical_trial",
                "condition": row.get("Conditions", ""),
                "timestamp": datetime.now().isoformat()
            }

            # Save to Memory
            hippocampus.save_memory(
                text=narrative,
                embedding=embedding,
                metadata=metadata
            )
            
            count += 1
            if count % 10 == 0:
                print(f"✅ Ingested {count} trials...")

    print(f"🎉 Finished! Total trials ingested: {count}")

if __name__ == "__main__":
    main()
