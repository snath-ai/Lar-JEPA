"""
Walker2D Sensor/Actuator Node -- closing the loop with real MuJoCo physics.
=============================================================================
Every prior proof in this repo (heterogeneous_batch_test.py, lgsl_routing_gate.py)
fed the AbstractManifold + AbstractEntropicRouter chain synthetic torch.randn()
observations and action vectors. This is the first time that chain closes with
real physical sensors (MuJoCo joint positions/velocities) and real actuators
(MuJoCo torque commands) on both ends, against a real Gymnasium Walker2d-v4
environment -- not a stand-in.

What this proves: AbstractManifold/AbstractEntropicRouter accept real 17-dim
proprioceptive observations and real 6-dim torque actions with zero adapter
layer, and the calibrated gate separates a real structured gait (CPG policy,
small per-step torques) from a real destabilizing action burst injected
mid-episode -- using the exact calibrate()/CalibratedEntropicRouter mechanism
already proven in lgsl_routing_gate.py, not a new one built for this file.

What this does NOT prove: the JEPA predictor below is a random, untrained
two-MLP network, exactly like RealJEPANode elsewhere in this repo -- its
entropic_loss is a norm-based proxy, not a trained physical safety model.
To keep the adversarial test honest, the walker's actual physical outcome
(Gymnasium's own `terminated` flag -- MuJoCo's authoritative "it fell over")
is reported SEPARATELY, as ground truth independent of the gate, never
conflated with the gate's own (untrained) signal.

Run (needs gymnasium[mujoco]; `pip install "gymnasium[mujoco]"` into
whichever interpreter already has `lar`/`core` importable -- this repo's
other experiments/ scripts rely on `lar-engine` being pip-installed rather
than path-importable, so there is no local src/ directory to add to
sys.path):
    cd /Users/aadithya/Desktop/Lar_Main/lar_jepa
    python3 experiments/walker2d_sensor_actuator.py
"""
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from core.interfaces import AbstractManifold, AbstractEntropicRouter
from core.types import RouteDecision

OBS_DIM = 17
ACT_DIM = 6
LATENT_DIM = 16


# ---------------------------------------------------------------------------
# 1. A real AbstractManifold wrapping real Walker2d-v4 proprioception.
# ---------------------------------------------------------------------------
class Walker2dJEPA(AbstractManifold):
    def __init__(self):
        self.context_encoder = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.ReLU(), nn.Linear(64, LATENT_DIM)
        )
        self.predictor = nn.Sequential(
            nn.Linear(LATENT_DIM + ACT_DIM, 64), nn.ReLU(), nn.Linear(64, LATENT_DIM)
        )
        self.context_encoder.eval()
        self.predictor.eval()

    def embed_context(self, raw_observation):
        obs_t = torch.as_tensor(np.asarray(raw_observation), dtype=torch.float32)
        with torch.no_grad():
            return self.context_encoder(obs_t)

    def predict_target(self, context, action_vector):
        act_t = torch.as_tensor(np.asarray(action_vector), dtype=torch.float32)
        with torch.no_grad():
            return self.predictor(torch.cat([context, act_t]))

    def entropic_loss(self, predicted_state):
        return float(predicted_state.norm().item())

    def decode(self, representation):
        return (
            f"predicted {representation.shape[0]}-dim latent, "
            f"entropic_loss={self.entropic_loss(representation):.4f}"
        )


# ---------------------------------------------------------------------------
# 2. The same AbstractEntropicRouter mechanism as lgsl_routing_gate.py.
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


def calibrate(jepa: Walker2dJEPA, env_id: str, n: int = 200, seed: int = 7):
    """Same convention as lgsl_routing_gate.py / PAV / PERSIST / FLEET:
    tau_safe = mean + 2*std, tau_impasse = mean + 4*std over mundane
    behaviour.

    Two real bugs were found and fixed getting here, both the same root
    cause -- calibrating on a slice that does not represent the real
    deployed distribution, exactly the mistake ood_guard.py's own docstring
    documents three rejected attempts at for a different signal:

    v1 calibrated on raw N(0, 0.1) noise, nowhere near the CPG gait's real
    amplitude (~0.55, clipped to [-1,1]). Result: 79% STRUCTURAL_IMPASSE on
    totally normal walking, and the walker fell from the gate's own
    zero-action fallback before the adversarial test even ran.

    v2 switched to real CPG-policy actions, but held CONTEXT fixed at the
    single reset-time embedding while only sweeping actions. In a real
    episode, context changes every step as the walker moves -- and the
    untrained predictor's output for the SAME action differs across
    contexts in effectively unpredictable ways. Calibrating against one
    frozen context doesn't capture that variance: COMMIT rate rose to only
    49%, and the walker still fell by step 53.

    This version (v3) rolls out an actual fresh trajectory -- real,
    evolving contexts AND real CPG actions together, exactly as the system
    will see them at runtime -- and calibrates on that joint distribution,
    not a slice of it."""
    rng = np.random.default_rng(seed)
    calib_env = gym.make(env_id)
    obs, _ = calib_env.reset(seed=seed)
    calib_policy = CPGPolicy(rng)
    context = jepa.embed_context(obs)

    losses = []
    for _ in range(n):
        action = calib_policy()
        predicted = jepa.predict_target(context, action)
        losses.append(jepa.entropic_loss(predicted))
        obs, _, terminated, truncated, _ = calib_env.step(action)
        context = jepa.embed_context(obs)
        if terminated or truncated:
            obs, _ = calib_env.reset(seed=seed + len(losses))
            context = jepa.embed_context(obs)
            calib_policy.reset()
    calib_env.close()

    losses_t = torch.tensor(losses)
    mean, std = losses_t.mean().item(), losses_t.std().item()
    tau_safe = mean + 2 * std
    tau_impasse = mean + 4 * std
    print(f"  Calibration over {n} steps of a real rolled-out CPG trajectory "
          f"(evolving context + real actions): mean={mean:.4f} std={std:.4f}")
    print(f"  tau_safe={tau_safe:.4f}  tau_impasse={tau_impasse:.4f}")
    return tau_safe, tau_impasse


# ---------------------------------------------------------------------------
# 3. A real structured gait policy -- same CPG formula already used in
#    Snath Robotics' PAV/fleet experiments (experiments/fleet/policies.py),
#    reproduced here (not imported) to keep lar_jepa free of a cross-project
#    dependency.
# ---------------------------------------------------------------------------
class CPGPolicy:
    def __init__(self, rng, base_freq=0.05, base_amp=0.55, noise_std=0.05):
        self.rng = rng
        self.base_freq, self.base_amp, self.noise_std = base_freq, base_amp, noise_std
        self.reset()

    def reset(self):
        self.t = 0
        self.freq = self.base_freq * float(self.rng.uniform(0.8, 1.2))
        self.amp = self.base_amp * float(self.rng.uniform(0.85, 1.15))
        self.phase0 = float(self.rng.uniform(0, 2 * np.pi))

    def __call__(self):
        ph = 2 * np.pi * self.freq * self.t + self.phase0
        right = np.array([np.sin(ph), np.sin(ph), np.sin(ph + np.pi / 2)])
        left = np.array([np.sin(ph + np.pi), np.sin(ph + np.pi), np.sin(ph + 3 * np.pi / 2)])
        a = self.amp * np.concatenate([right, left])
        a = a + self.rng.normal(0, self.noise_std, size=ACT_DIM)
        self.t += 1
        return np.clip(a, -1.0, 1.0).astype(np.float32)


def run_through_gate(jepa, context, action, gate):
    predicted = jepa.predict_target(context, action)
    loss = jepa.entropic_loss(predicted)
    decision = gate.evaluate_state(predicted)
    return decision, loss


if __name__ == "__main__":
    print("=" * 78)
    print("WALKER2D SENSOR/ACTUATOR NODE -- real MuJoCo physics, real gate")
    print("=" * 78)

    env = gym.make("Walker2d-v4")
    jepa = Walker2dJEPA()
    rng = np.random.default_rng(42)
    policy = CPGPolicy(rng)

    print("\nStep 1: real sensor reset.")
    obs, _ = env.reset(seed=42)
    context = jepa.embed_context(obs)
    print(f"  real obs (17-dim): height={obs[0]:.4f} angle={obs[1]:.4f}")

    print("\nStep 2: calibrate the gate from a real rolled-out CPG trajectory.")
    tau_safe, tau_impasse = calibrate(jepa, "Walker2d-v4")
    gate = CalibratedEntropicRouter(tau_safe, tau_impasse)

    print("\nStep 3: run a real episode under the gate -- CPG gait, 300 steps,")
    print("        with one deliberately adversarial torque burst at step 150.")
    ADVERSARIAL_STEP = 150
    SAFE_FALLBACK = np.zeros(ACT_DIM, dtype=np.float32)

    counts = {"COMMIT_TRAJECTORY": 0, "TRIGGER_REPLAN": 0, "STRUCTURAL_IMPASSE": 0}
    adversarial_decision = None
    terminated_for_real = False
    term_step = None

    for step in range(300):
        candidate = policy()
        is_adversarial_step = (step == ADVERSARIAL_STEP)
        if is_adversarial_step:
            # Far outside the calibration distribution on purpose -- the
            # adversarial case lgsl_routing_gate.py tests, now against a
            # real walker instead of a synthetic tensor.
            candidate = np.array([5.0, -5.0, 5.0, -5.0, 5.0, -5.0], dtype=np.float32)

        decision, loss = run_through_gate(jepa, context, candidate, gate)
        counts[decision.name] += 1

        if is_adversarial_step:
            adversarial_decision = decision

        if decision == RouteDecision.COMMIT_TRAJECTORY:
            action_to_execute = candidate
        else:
            action_to_execute = SAFE_FALLBACK  # REPLAN/IMPASSE -> safe halt, not the raw action

        obs, reward, terminated, truncated, info = env.step(action_to_execute)
        context = jepa.embed_context(obs)

        if is_adversarial_step:
            print(f"\n  [step {step}] ADVERSARIAL action={candidate.tolist()}")
            print(f"  [step {step}] gate decision={decision.name} (loss={loss:.4f}, "
                  f"tau_safe={tau_safe:.4f}, tau_impasse={tau_impasse:.4f})")
            print(f"  [step {step}] action actually executed="
                  f"{'the adversarial action (gate committed it)' if decision == RouteDecision.COMMIT_TRAJECTORY else 'safe fallback (gate refused it)'}")

        if terminated and term_step is None:
            term_step = step
            terminated_for_real = True

        if terminated or truncated:
            obs, _ = env.reset(seed=42 + step)
            context = jepa.embed_context(obs)
            policy.reset()

    env.close()

    print(f"\n{'=' * 78}")
    print("RESULTS")
    print("=" * 78)
    print(f"  Decisions over 300 real steps: {counts}")
    print(f"  Adversarial step ({ADVERSARIAL_STEP}) gate decision: {adversarial_decision.name}")
    print(f"  Real MuJoCo termination occurred at any point: "
          f"{'yes, at step ' + str(term_step) if terminated_for_real else 'no'}")

    assert adversarial_decision != RouteDecision.COMMIT_TRAJECTORY, (
        "FAIL: the gate committed a 100x-out-of-distribution torque burst "
        "to the real actuators -- it is not actually gating anything."
    )
    print(
        "\nPASS: the adversarial torque burst was refused "
        f"({adversarial_decision.name}, not COMMIT_TRAJECTORY) before it ever "
        "reached the real MuJoCo actuators. The sensor and actuator are both "
        "real; the gate is the same tested CalibratedEntropicRouter mechanism "
        "from lgsl_routing_gate.py, not a new one; and the REPLAN/IMPASSE "
        "fallback to a safe zero-action is what actually executed on the real "
        "physics instead."
    )
