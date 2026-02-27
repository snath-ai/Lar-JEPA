<div align="center">

# Lar-JEPA: Orchestrating World Models (A Post-LLM Architecture)
**A deterministic execution spine and cognitive memory layer for Predictive Architectures.**

<p align="center">
  <a href="https://github.com/snath-ai/lar">
    <img alt="Based on" src="https://img.shields.io/badge/Variant%20of-Lár%20Engine-blue?style=for-the-badge">
  </a>
  <a href="https://github.com/snath-ai/Lar-JEPA">
    <img alt="Architecture" src="https://img.shields.io/badge/Architecture-Predictive%20World%20Models-blueviolet?style=for-the-badge">
  </a>
</p>

</div>

---

## The Autoregressive Bottleneck
For the past three years, AI research has focused heavily on **Prompt Engineering** autoregressive Large Language Models (LLMs) to create autonomous agents. Frameworks like LangChain, AutoGPT, and ReAct operate on a fundamental assumption: *The mind of the agent is a sequence of conversational text.*

**This approach is hitting a wall.**
When an LLM agent hallucinates a poorly reasoned action on step 3 of a 50-step plan, the entire execution is doomed. Why? Because its "memory" is simply a massive, linear string of text appended to a context window. LLMs do not inherently understand physics, spatial logic, or long-term consequence; they predict the most statistically probable next token. 

## The Post-LLM Paradigm (JEPA)
Yann LeCun’s **Joint Embedding Predictive Architecture (JEPA)** offers a profound paradigm shift. 
JEPAs (like V-JEPA or I-JEPA) do not predict the next word; they predict the abstract *state of the world*. They learn to ignore irrelevant background noise and focus on the conceptual mechanics of a problem. 

A JEPA allows an agent to imagine different actions and predict the environmental outcomes in a hidden mathematical representation space (latent tensors) *before* committing to a physical action.

**The Orchestration Problem:**
How do you route an abstract mathematical space? 
Current orchestration frameworks are entirely text-dependent. If your state-of-the-art World Model outputs a 768-dimensional tensor symbolizing a "crash," traditional frameworks crash.
**You need a new Nervous System.**

---

## The Trio Architecture
This repository contains the three pillars of a true Cognitive Architecture, designed specifically for researchers building non-LLM, predictive agents:

### 1. The World Simulator (JEPA)
The "Imagination." This model predicts possible future states. It plans the best path to a goal by hallucinating consequences in a latent mathematical space, completely bypassing token-by-token text generation.

### 2. The Execution Spine ([Lár](https://github.com/snath-ai/lar))
[Lár](https://github.com/snath-ai/lar) is a deterministic, topological DAG (Directed Acyclic Graph) framework, originally designed as "PyTorch for Agents." 
Unlike text-based frameworks, Lár passes a flexible `GraphState`. It can seamlessly pass massive, dense latent tensors (the JEPA predictions) through strict `RouterNodes`. Lár evaluates the mathematical danger or reward of a future state and reroutes the execution flow *before* the action hits the real world.

### 3. The Cognitive Memory (DMN)
The **Default Mode Network (DMN)** provides episodic memory and solves catastrophic forgetting. It watches the Lár execution logs in the background. When the agent "sleeps," the DMN scans the day's successes and failures. It consolidates those expensive, slow JEPA simulations into cheap, permanent "muscle memory" heuristics—meaning the agent continually learns from its environment without requiring immediate retraining.

---

## Why Lár is the Premier Framework for World Models
Most agentic frameworks (LangChain, AutoGPT, CrewAI) were built for chatbots. They rely on prompting LLMs to output conversational text and parsing those strings to decide what to do next. 

If you are training World Models (like JEPAs), your model doesn't output text—it outputs **high-dimensional mathematical tensors** representing the abstract state of a physical environment. 

**Lár is structurally superior for World Models because:**
1. **Mathematical Routing (No Prompting):** You don't prompt Lár. You write deterministic Python `RouterNodes` that evaluate the latent tensors directly (e.g., `if collision_probability > 0.85: return "REPLAN"`).
2. **Native Tensor Logging:** The `AuditLogger` in this repository has been custom-patched with a `TensorSafeEncoder`. You can pass massive PyTorch/Numpy tensors natively through the execution graph, and the Logger will gracefully serialize them into metadata (`{ "__type__": "Tensor", "shape": [1, 768] }`) instead of crashing the JSON stringifier.
3. **Out-of-the-Box Determinism:** Plug your JEPA directly into a Lár `PredictiveNode`. Let the framework handle state transport, error-handling, and safety rollbacks without LLM hallucinations breaking the loop.
4. **System 1 / System 2 Testing:** Formally measure the difference between fast-reflex execution (System 1) and deep-simulation planning (System 2) using built-in `RouterNodes`.
5. **Continuous Learning:** Use the DMN "Sleep Cycle" architecture to publish and iterate on long-term heuristic consolidation.

---

##  The Proof of Concept
Run the standalone simulation today to see Lár orchestrating a conceptual world model without generating a single word of English text. 

*In this simulation, a mock JEPA evaluates accelerating toward a wall. The Lár System 2 Router analyzes the predicted numerical state, detects the impending collision locally, vetoes the action, triggers a replanning phase, and safely halts the robot.*

```bash
cd lar_jepa/lar
poetry install
poetry run python examples/advanced/13_world_model_jepa.py
```

### The Output Trace:
```
==================================================
 LÁR JEPA WORLD MODEL DEMO
==================================================
  [JEPA] Current World State: [X=8, V=5]
  [JEPA] Simulating Action: 'ACCELERATE'...
  [JEPA] Predicted Future State: {'predicted_x': 18, 'predicted_v': 10}
  
  [System 2 Router] CRASH DETECTED in simulation (Hit Wall at X=10). Vetoing action.
  [RouterNode]: Decision function returned 'REPLAN_NODE'
  [RouterNode]: Routing to PredictiveJepaNode
  
  [JEPA] Current World State: [X=8, V=5]
  [JEPA] Simulating Action: 'BRAKE'...
  [JEPA] Predicted Future State: {'predicted_x': 8, 'predicted_v': 0}
  
  [System 2 Router] Simulation Safe. Action Approved.
  [RouterNode]: Decision function returned 'EXECUTE_NODE'
  [RouterNode]: Routing to PhysicalMotorNode
  
  [MOTOR] Executing physical movement! Applying simulated state to reality.
✅ Notice how the Agent avoided the crash entirely by running an internal world model.
```

---
**Bet on Architecture, not just Compute.** 
The industry is building the Brain (JEPAs). We are building the Nervous System (Lár + DMN).

---

## License
**Apache 2.0**.

**Note**: This repository is a **Conceptual Testbed and Showcase** built upon the [Lár Engine](https://github.com/snath-ai/lar). It is intended to demonstrate advanced concepts in Predictive World Models, Mathematical Routing, and Post-LLM orchestration.
