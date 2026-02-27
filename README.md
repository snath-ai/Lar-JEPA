<div align="center">

# Lár-JEPA: Orchestrating World Models (A Post-LLM Architecture)
**A deterministic execution spine and cognitive memory layer for Predictive Architectures.**

</div>

---

## The Autoregressive Bottleneck
The AI industry is obsessed with trying to prompt-engineer text-predicting LLMs into becoming autonomous agents. But chatbots are not agents, and predicting the next token is not general intelligence.

When an LLM hallucinated a bad action on step 3 of a 50-step plan, the entire execution is doomed because its "memory" is just a massive, linear string of text shoved into a context window.

## The Post-LLM Paradigm (JEPA)
Yann LeCun’s **JEPA (Joint Embedding Predictive Architecture)** solves this. JEPAs do not predict the next word; they predict the abstract *state of the world*. They learn to ignore irrelevant noise and focus on the conceptual mechanics of a problem. They allow an agent to imagine different actions and predict the outcomes in a hidden mathematical representation space *before* acting physically.

But how do you route an abstract mathematical space? 
**You need a Nervous System.**

---

## The Trio Architecture
This repository contains the three pillars of a true Cognitive Architecture:

1. **The World Simulator (JEPA):** Predicts possible futures and plans the best path to a goal without getting bogged down in token-by-token generation.
2. **The Execution Spine (Lár):** A deterministic, topological DAG framework. It passes the abstract latent tensors (the JEPA predictions) through strict `RouterNodes`, evaluating the mathematical danger of a future state and adapting *before* the action hits the real world.
3. **The Cognitive Memory (DMN):** The Default Mode Network. It watches the Lár execution logs all day. When the agent "sleeps," DMN consolidates those expensive JEPA simulations into cheap, permanent "muscle memory" heuristics, solving catastrophic forgetting.

## The Proof of Concept
Run the standalone simulation today to see Lár orchestrating a world model without generating a single word of English text:

```bash
cd lar_jepa
poetry run python examples/advanced/13_world_model_jepa.py
```

### The Output:
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
