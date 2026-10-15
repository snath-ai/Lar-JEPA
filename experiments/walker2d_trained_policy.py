"""
Walker2D Sensor/Actuator Node -- with a TRAINED WALKING POLICY.
=============================================================================
Both prior Walker2d files (walker2d_sensor_actuator.py, walker2d_trained_jepa.py)
used a scripted central-pattern-generator gait as the action source, and both
found the walker falls at step 44-45 regardless of which JEPA variant gates
it -- confirmed with a no-gate baseline that falls at the identical step.
That isolated the fall to the POLICY, not the gate or the JEPA. This file
changes exactly that one variable: a real PPO policy, trained from scratch on
real Walker2d-v5 rollouts (stable-baselines3, 1.5M timesteps, ~11 minutes on
this machine -- see train_walker2d_ppo in this file), in place of the CPG
gait. Everything else -- the untrained AbstractManifold, the
CalibratedEntropicRouter mechanism, the trajectory-based calibration, the
adversarial-burst test at step 150 -- is identical to walker2d_sensor_actuator.py.

What this isolates: whether the earlier "walker falls early" finding was a
property of the gait policy specifically (expected: a trained policy survives
far longer) or something deeper in the loop (expected: if a trained policy
ALSO fails fast, that would implicate the gate or the JEPA after all, and
would need investigating rather than asserting).

Run:
    cd /Users/aadithya/Desktop/Lar_Main/lar_jepa
    python3 experiments/walker2d_trained_policy.py
"""
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from core.types import RouteDecision
from walker2d_sensor_actuator import Walker2dJEPA, CalibratedEntropicRouter

ACT_DIM = 6
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "models", "walker2d_ppo.zip")
ENV_ID = "Walker2d-v5"


def train_walker2d_ppo(total_timesteps: int = 1_500_000, seed: int = 42):
    """Real PPO training, real Walker2d-v5 physics, no shortcuts. Saved so
    this only has to run once."""
    print(f"  Training PPO from scratch: {total_timesteps} real timesteps "
          f"on {ENV_ID} ...")
    env = Monitor(gym.make(ENV_ID))
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    print(f"  Trained in {time.time() - t0:.1f}s")
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    model.save(CHECKPOINT.replace(".zip", ""))
    print(f"  Saved -> {CHECKPOINT}")
    return model


def load_or_train_policy():
    if os.path.exists(CHECKPOINT):
        print(f"  Loading existing checkpoint from {CHECKPOINT}")
        return PPO.load(CHECKPOINT)
    return train_walker2d_ppo()


def calibrate(jepa, policy, env_id, n=200, seed=7):
    """Same v3 methodology as walker2d_sensor_actuator.py: real rolled-out
    trajectory, evolving context AND real (now policy-driven, not scripted)
    actions together."""
    calib_env = gym.make(env_id)
    obs, _ = calib_env.reset(seed=seed)
    context = jepa.embed_context(obs)

    losses = []
    for _ in range(n):
        action, _ = policy.predict(obs, deterministic=True)
        predicted = jepa.predict_target(context, action)
        losses.append(jepa.entropic_loss(predicted))
        obs, _, terminated, truncated, _ = calib_env.step(action)
        context = jepa.embed_context(obs)
        if terminated or truncated:
            obs, _ = calib_env.reset(seed=seed + len(losses))
            context = jepa.embed_context(obs)
    calib_env.close()

    losses_t = torch.tensor(losses)
    mean, std = losses_t.mean().item(), losses_t.std().item()
    tau_safe, tau_impasse = mean + 2 * std, mean + 4 * std
    print(f"  Calibration over {n} real policy-driven steps: mean={mean:.4f} std={std:.4f}")
    print(f"  tau_safe={tau_safe:.4f}  tau_impasse={tau_impasse:.4f}")
    return tau_safe, tau_impasse


if __name__ == "__main__":
    print("=" * 78)
    print("WALKER2D SENSOR/ACTUATOR NODE -- with a TRAINED WALKING POLICY")
    print("=" * 78)

    print("\nStep 0: load or train the real PPO walking policy.")
    policy = load_or_train_policy()

    jepa = Walker2dJEPA()

    print("\nStep 1: real sensor reset.")
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=42)
    context = jepa.embed_context(obs)

    print("\nStep 2: calibrate the gate from a real rolled-out trajectory "
          "under the TRAINED policy (not CPG).")
    tau_safe, tau_impasse = calibrate(jepa, policy, ENV_ID)
    gate = CalibratedEntropicRouter(tau_safe, tau_impasse)

    print("\nStep 3: run a real episode under the gate -- trained policy, up to")
    print("        1000 steps, with one adversarial torque burst at step 150")
    print("        (only if the walker is still standing by then).")
    ADVERSARIAL_STEP = 150
    SAFE_FALLBACK = np.zeros(ACT_DIM, dtype=np.float32)
    N_STEPS = 1000

    counts = {"COMMIT_TRAJECTORY": 0, "TRIGGER_REPLAN": 0, "STRUCTURAL_IMPASSE": 0}
    adversarial_decision = None
    terminated_for_real, term_step = False, None

    for step in range(N_STEPS):
        candidate, _ = policy.predict(obs, deterministic=True)
        candidate = np.asarray(candidate, dtype=np.float32)
        is_adversarial_step = (step == ADVERSARIAL_STEP)
        if is_adversarial_step:
            candidate = np.array([5.0, -5.0, 5.0, -5.0, 5.0, -5.0], dtype=np.float32)

        predicted = jepa.predict_target(context, candidate)
        loss = jepa.entropic_loss(predicted)
        decision = gate.evaluate_state(predicted)
        counts[decision.name] += 1
        if is_adversarial_step:
            adversarial_decision = decision
            print(f"\n  [step {step}] ADVERSARIAL action={candidate.tolist()}")
            print(f"  [step {step}] gate decision={decision.name} (loss={loss:.4f}, "
                  f"tau_safe={tau_safe:.4f}, tau_impasse={tau_impasse:.4f})")

        action_to_execute = candidate if decision == RouteDecision.COMMIT_TRAJECTORY else SAFE_FALLBACK
        obs, reward, terminated, truncated, info = env.step(action_to_execute)
        context = jepa.embed_context(obs)

        if terminated and term_step is None:
            term_step = step
            terminated_for_real = True
        if terminated or truncated:
            # do not reset -- let the episode end here, matching how a real
            # deployed walker would not get a free respawn after falling
            break
    env.close()

    print(f"\n{'=' * 78}")
    print("RESULTS -- trained walking policy vs. scripted CPG gait")
    print("=" * 78)
    steps_survived = step + 1
    print(f"  Decisions over {steps_survived} real steps: {counts}")
    print(f"  Walker survived: {steps_survived} steps "
          f"{'(fell, terminated at step ' + str(term_step) + ')' if terminated_for_real else '(did not fall within ' + str(N_STEPS) + ' steps)'}")
    print(f"  For comparison: the CPG gait (walker2d_sensor_actuator.py) fell at "
          f"step 45, and a no-gate CPG baseline fell at the identical step 45.")
    if adversarial_decision is not None:
        print(f"  Adversarial step ({ADVERSARIAL_STEP}) gate decision: {adversarial_decision.name}")
        assert adversarial_decision != RouteDecision.COMMIT_TRAJECTORY, (
            "FAIL: the gate committed a 100x-out-of-distribution torque burst "
            "to the real actuators even with a trained policy's calibration."
        )
        print(
            "\nPASS: adversarial burst refused "
            f"({adversarial_decision.name}) with a trained walking policy too."
        )
    else:
        print(f"\n  Walker fell before step {ADVERSARIAL_STEP} -- the adversarial "
              f"test never ran. This itself is the honest result: see above.")
