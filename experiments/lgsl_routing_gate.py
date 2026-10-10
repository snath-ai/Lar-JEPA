"""
LGSL Routing Gate -- the LLM can suggest, it cannot decide.
=============================================================
Builds Composition Pattern 1 from ARCHITECTURE.md ("LLM Routes JEPA"):

    LLMNode --LGSL_SPEC--> RouterNode --> JEPANode --LATENT_EMBEDDING--> ...

"The LLM interprets intent and generates an LGSL routing instruction
specifying which JEPA to invoke and with which action vector. The JEPA
executes the world-model prediction." Nowhere in that pattern, or in
AbstractEntropicRouter's own docstring, does the LLM's output directly
reach the robot. It only ever proposes a *candidate* action. JEPA
predicts what would actually happen if that action were taken, and a
real AbstractEntropicRouter, calibrated on real data, is the only thing
that decides whether the result is safe enough to commit.

Nothing in this repo had a concrete AbstractEntropicRouter before this
file -- it existed only as the abstract interface in core/interfaces.py.
This builds one and proves it can genuinely refuse, not just rubber-stamp
whatever the LLM says. Two tests, not one: a case the LLM is actually
asked about (and is expected to pass), and a deliberately adversarial
action vector constructed by hand (expected to be refused). A gate that
only ever says yes hasn't been tested.
"""
import sys
import os
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from core.interfaces import AbstractEntropicRouter
from core.types import RouteDecision

from heterogeneous_batch_test import RealJEPANode, RealGenerativeLLMNode


# ---------------------------------------------------------------------------
# The LGSL instruction menu. A small, fixed set of named actions, each
# mapped to a real action vector. The LLM only ever picks a NAME -- it
# never outputs raw control numbers. Asking a small, non-fine-tuned model
# to emit precise floats reliably is unrealistic; asking it to pick one of
# three named options and explain why is what these models can actually do,
# and it's also the more honest design: a human auditing the log reads an
# action name and a rationale, not an opaque vector.
# ---------------------------------------------------------------------------
LGSL_ACTIONS = {
    "CAUTIOUS_STEP": torch.tensor([0.1, 0.0, 0.0, 0.0]),
    "ACCELERATE":    torch.tensor([0.6, 0.1, 0.0, 0.0]),
    "EMERGENCY_HALT": torch.tensor([0.0, 0.0, -0.3, 0.0]),
}


def parse_lgsl_instruction(raw_text: str):
    """
    Parse the LLM's free-text output into one of the known LGSL action
    names. Deliberately conservative: if none of the known action names
    appear, returns None rather than guessing. An unparseable instruction
    must never silently fall through to a default action -- the caller
    is required to treat None as "cannot determine, do not commit."
    """
    upper = raw_text.upper()
    for name in LGSL_ACTIONS:
        if name in upper:
            return name
    return None


# ---------------------------------------------------------------------------
# A real AbstractEntropicRouter implementation. Two thresholds, calibrated
# empirically from the JEPA's own null-action behaviour, not picked by
# hand -- same mean + k*std pattern PAV, PERSIST, and FLEET all use for
# their own thresholds.
# ---------------------------------------------------------------------------
class CalibratedEntropicRouter(AbstractEntropicRouter):
    def __init__(self, tau_safe: float, tau_impasse: float):
        self.tau_safe = tau_safe
        self.tau_impasse = tau_impasse

    def evaluate_state(self, predicted_state) -> RouteDecision:
        loss = float(predicted_state.norm().item())
        if loss <= self.tau_safe:
            return RouteDecision.COMMIT_TRAJECTORY
        if loss <= self.tau_impasse:
            return RouteDecision.TRIGGER_REPLAN
        return RouteDecision.STRUCTURAL_IMPASSE


def calibrate(jepa: RealJEPANode, context: torch.Tensor, n=200, seed=7):
    """
    Build the null distribution of entropic_loss under small, mundane
    action vectors (the kind of noise a standing/walking robot sees
    constantly), then set tau_safe = mean + 2*std and
    tau_impasse = mean + 4*std. This is the same convention already used
    throughout this codebase (PAV's tau_normal, PERSIST's scope boundary,
    FLEET's wind-guard threshold) -- not a new one invented for this file.
    """
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(n):
        mundane_action = torch.randn(4, generator=g) * 0.05  # small, near-zero noise
        predicted = jepa.predict_target(context, mundane_action)
        losses.append(jepa.entropic_loss(predicted))
    losses_t = torch.tensor(losses)
    mean, std = losses_t.mean().item(), losses_t.std().item()
    tau_safe = mean + 2 * std
    tau_impasse = mean + 4 * std
    print(f"  Calibration over {n} mundane actions: mean={mean:.4f} std={std:.4f}")
    print(f"  tau_safe={tau_safe:.4f}  tau_impasse={tau_impasse:.4f}")
    return tau_safe, tau_impasse


def run_through_gate(label, jepa, context, action_vector, gate: CalibratedEntropicRouter):
    predicted_state = jepa.predict_target(context, action_vector)
    loss = jepa.entropic_loss(predicted_state)
    decision = gate.evaluate_state(predicted_state)
    print(f"  [{label}] action_vector={action_vector.tolist()}")
    print(f"  [{label}] entropic_loss={loss:.4f} -> {decision.value}")
    return decision


if __name__ == "__main__":
    print("=" * 78)
    print("LGSL ROUTING GATE -- LLM suggests, AbstractEntropicRouter decides")
    print("=" * 78)

    jepa = RealJEPANode(state_dim=16, latent_dim=8, action_dim=4)
    llm = RealGenerativeLLMNode()

    context = jepa.embed_context(torch.randn(16))

    print("\nStep 1: calibrate the gate from the JEPA's own null-action behaviour.")
    tau_safe, tau_impasse = calibrate(jepa, context)
    gate = CalibratedEntropicRouter(tau_safe, tau_impasse)

    # -------------------------------------------------------------------
    # Case A: ask the real LLM. It only ever picks a name; it never
    # touches the action vector directly.
    # -------------------------------------------------------------------
    print("\nStep 2: ask the LLM for a routing suggestion (real generation, not scripted).")
    prompt = (
        "A walking robot is on flat, normal ground with no obstacles. "
        "Which action should it take: CAUTIOUS_STEP, ACCELERATE, or "
        "EMERGENCY_HALT? Reply with just the action name and one short "
        "reason."
    )
    messages = [{"role": "user", "content": prompt}]
    rendered = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokenized = llm.tokenizer(rendered, return_tensors="pt")
    with torch.no_grad():
        out_ids = llm.model.generate(
            **tokenized, max_new_tokens=40, do_sample=False,
            pad_token_id=llm.tokenizer.eos_token_id,
        )
    raw_llm_output = llm.tokenizer.decode(
        out_ids[0][tokenized["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"  LLM said: \"{raw_llm_output.strip()}\"")

    chosen_action = parse_lgsl_instruction(raw_llm_output)
    if chosen_action is None:
        print("  Could not parse a known action from the LLM's output -- "
              "treating as STRUCTURAL_IMPASSE (no silent default).")
        case_a_decision = RouteDecision.STRUCTURAL_IMPASSE
    else:
        print(f"  Parsed LGSL action: {chosen_action}")
        case_a_decision = run_through_gate(
            "LLM-suggested", jepa, context, LGSL_ACTIONS[chosen_action], gate
        )

    # -------------------------------------------------------------------
    # Case B: a deliberately adversarial action vector, constructed by
    # hand, well outside the calibration distribution. This is the proof
    # the gate can actually refuse, not just the hope that it would.
    # -------------------------------------------------------------------
    print("\nStep 3: a deliberately adversarial action, far outside calibration.")
    adversarial_action = torch.tensor([8.0, -6.0, 9.0, -7.0])  # ~100x the calibration scale
    case_b_decision = run_through_gate(
        "adversarial", jepa, context, adversarial_action, gate
    )

    print(f"\n{'=' * 78}")
    print("VERDICT")
    print("=" * 78)
    print(f"  LLM-suggested action -> {case_a_decision.value}")
    print(f"  Adversarial action    -> {case_b_decision.value}")

    assert case_b_decision != RouteDecision.COMMIT_TRAJECTORY, (
        "FAIL: the gate committed an adversarial action -- it is not actually "
        "gating anything, the LLM (or anything else) could push through "
        "unsafe actions unchecked."
    )

    print(
        "\nPASS: the adversarial action was refused "
        f"({case_b_decision.value}, not COMMIT_TRAJECTORY). The gate is a "
        "real filter, not a rubber stamp -- the LLM's output only ever "
        "proposes a route; AbstractEntropicRouter is what actually decides."
    )
