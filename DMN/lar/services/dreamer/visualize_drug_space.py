import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re

# Configuration
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "../../memory/dreams.json")
OUTPUT_PLOT = os.path.join(os.path.dirname(__file__), "drug_space_gap_analysis.png")

def load_trials():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def extract_intervention(text):
    # Try to grab the "X tested Y" part for labels
    try:
        # Heuristic: usually "Study [Title] tested [Intervention] on"
        part = text.split("tested ")[1].split(" on ")[0]
        # Clean
        part = part.replace("DRUG: ", "").replace("RADIATION: ", "").replace("|", "+")
        # Truncate
        return part[:20] + "..." if len(part) > 20 else part
    except:
        return "Unknown"

def main():
    print("🧠 Loading Hippocampus memory...")
    trials = load_trials()
    if not trials:
        print("Empty memory.")
        return

    print(f"📊 Processing {len(trials)} trials...")
    
    # 1. Prepare Text Data (Corpus)
    # We use the full narrative for the vector/position, but intervention for label
    corpus = [t.get("content", "") for t in trials]
    labels = [extract_intervention(t) for t in trials]
    statuses = [t.get("metadata", {}).get("status", "UNKNOWN") for t in trials]

    # 2. Vectorization (TF-IDF) - Zero Compute Approximation of Semantic Space
    print("🧮 Generating Vector Space (TF-IDF + PCA)...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(corpus)

    # 3. Dimensionality Reduction (PCA -> 2D)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.toarray())

    # 4. Plotting
    print(f"🎨 Painting Drug Space to {OUTPUT_PLOT}...")
    plt.figure(figsize=(12, 10))
    
    # Color code by status
    colors = []
    sizes = []
    for s in statuses:
        if s == "COMPLETED":
            colors.append("blue")
            sizes.append(50)
        elif s in ["TERMINATED", "SUSPENDED", "WITHDRAWN"]:
            colors.append("red")
            sizes.append(100) # Highlight failures
        else:
            colors.append("gray") # Recruitment etc
            sizes.append(30)

    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, alpha=0.6)

    # Label a subset to avoid clutter (e.g., failures and some successes)
    for i, (x, y) in enumerate(coords):
        if statuses[i] in ["TERMINATED", "WITHDRAWN"] or i % 10 == 0:
            plt.text(x, y, labels[i], fontsize=8, alpha=0.8)

    plt.title("Clinical Trial 'Gap Analysis' (Blue=Complete, Red=Failed, Gray=Ongoing)")
    plt.xlabel("Principal Component 1 (Trial Composition)")
    plt.ylabel("Principal Component 2 (Intervention Type)")
    plt.grid(True, linestyle= '--', alpha=0.5)
    
    # Annotate "The Gap" (Hypothetical empty space)
    # Just draw a circle in a sparse area? No, let's just let the user see it.
    
    plt.savefig(OUTPUT_PLOT)
    print("✅ Done.")

if __name__ == "__main__":
    main()
