"""
Walker2D Sensor/Actuator Node -- with the ACTUALLY TRAINED encoder.
=============================================================================
walker2d_sensor_actuator.py proved the real-physics loop closes with an
untrained, randomly-initialised AbstractManifold (exactly like RealJEPANode
elsewhere in this repo). This file swaps in a genuinely trained JEPA
encoder: proj_jepa.pt, trained by the Snath Robotics project's own
train_jepa_walker2d.py on real Walker2d-v5 rollouts with a JEPA cosine loss
+ VICReg collapse-prevention term -- no labels, no mock weights.

Two honest caveats, found by reading the training script, not assumed:

1. The checkpoint only saves the ENCODER's state_dict. The JEPAPredictor
   that was trained alongside it (z_t -> z_{t+1}) was never checkpointed --
   a real gap in the original training script. This file trains a fresh
   predictor head on top of the FROZEN trained encoder (same cosine-loss
   objective, same real rollout data, ~3 minutes), rather than silently
   using an untrained predictor and calling the whole thing "trained."
   The encoder is genuinely pretrained; the predictor is freshly trained
   here, specifically for this file, on top of it.

2. The checkpoint's own self-reported verification (train_jepa_walker2d.py
   --epochs 60 --steps 80000, see `stats` in the .pt file) shows a thin
   separation margin: within-cluster cosine distance 0.82 for both normal
   and ice, between-cluster 0.84 -- a separation margin of only 0.017 and a
   centroid angle of 26.2 degrees. This is a real, modestly-trained encoder,
   not a strongly discriminative one. Reported here, not hidden.

Cross-project dependency: this file imports the real ProprioceptiveEncoder
class directly from the Snath Robotics project (not reimplemented here,
to guarantee the checkpoint's state_dict loads correctly against the exact
real architecture it was trained with -- a reimplementation risks a subtle
mismatch that load_state_dict would not necessarily catch). Run with the
Snath Robotics .venv, which has gymnasium[mujoco] and torch already:

    cd "/Users/aadithya/Desktop/Private/personal/Snath/JEPA_Playground/Snath Robotics"
    .venv/bin/python /Users/aadithya/Desktop/Lar_Main/lar_jepa/experiments/walker2d_trained_jepa.py
"""
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

_LAR_JEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SNATH_ROBOTICS_ROOT = (
    "/Users/aadithya/Desktop/Private/personal/Snath/JEPA_Playground/Snath Robotics"
)
# lar_jepa's own `core` package must win over Snath Robotics' differently-shaped
# `core` package (same name, different contents -- a real namespace collision
# found by hitting it) -- insert lar_jepa LAST so it sits at sys.path[0].
sys.path.insert(0, _SNATH_ROBOTICS_ROOT)
sys.path.insert(0, _LAR_JEPA_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from core.interfaces import AbstractManifold
from core.types import RouteDecision
from encoders.robotics.proprio_encoder import ProprioceptiveEncoder

from walker2d_sensor_actuator import CalibratedEntropicRouter, CPGPolicy

CHECKPOINT = os.path.join(_SNATH_ROBOTICS_ROOT, "models", "pav", "proprio_jepa.pt")
IMU_DIM, TACTILE_DIM, EMBED_DIM = 64, 32, 8
ACT_DIM = 6


def obs_to_encoder(obs: np.ndarray):
    """Identical mapping to train_jepa_walker2d.py's obs_to_encoder -- using
    a different one here would silently feed the trained weights data they
    were never trained on."""
    imu = torch.zeros(1, IMU_DIM)
    tactile = torch.zeros(1, TACTILE_DIM)
    t = torch.from_numpy(obs).float()
    imu[0, :17] = t
    tactile[0, :6] = t[2:8]
    tactile[0, 6:12] = t[11:17]
    return imu, tactile


class PredictorHead(nn.Module):
    """z_t -> z_{t+1} given action, trained fresh here (frozen encoder)."""
    def __init__(self, embed_dim=EMBED_DIM, act_dim=ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + act_dim, 32), nn.GELU(), nn.Linear(32, embed_dim)
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class TrainedWalker2dJEPA(AbstractManifold):
    def __init__(self, encoder: ProprioceptiveEncoder, predictor: PredictorHead):
        self.encoder = encoder
        self.predictor = predictor
        self.encoder.eval()
        self.predictor.eval()

    def embed_context(self, raw_observation):
        imu, tactile = obs_to_encoder(np.asarray(raw_observation))
        with torch.no_grad():
            return self.encoder(imu, tactile).squeeze(0)

    def predict_target(self, context, action_vector):
        act_t = torch.as_tensor(np.asarray(action_vector), dtype=torch.float32)
        with torch.no_grad():
            return self.predictor(context.unsqueeze(0), act_t.unsqueeze(0)).squeeze(0)

    def entropic_loss(self, predicted_state):
        return float(predicted_state.norm().item())

    def decode(self, representation):
        return f"predicted {representation.shape[0]}-dim latent (trained encoder)"


def train_predictor_head(encoder, env_id="Walker2d-v5", n_steps=4000, epochs=15, seed=0):
    """Train only the predictor, encoder frozen, on real rollouts -- the
    piece the original checkpoint never saved."""
    print(f"  Collecting {n_steps} real rollout steps on {env_id} (random actions, "
          f"matching the original training script's protocol) ...")
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    pairs = []
    for _ in range(n_steps):
        action = rng.uniform(-0.4, 0.4, size=ACT_DIM).astype(np.float32)
        obs_t = obs.copy()
        obs_next, _, terminated, truncated, _ = env.step(action)
        pairs.append((obs_t, action, obs_next.copy()))
        obs = obs_next
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    predictor = PredictorHead()
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    encoder.eval()

    with torch.no_grad():
        Z, A, Z1 = [], [], []
        for obs_t, action, obs_t1 in pairs:
            imu, tac = obs_to_encoder(obs_t)
            imu1, tac1 = obs_to_encoder(obs_t1)
            Z.append(encoder(imu, tac).squeeze(0))
            A.append(torch.from_numpy(action))
            Z1.append(encoder(imu1, tac1).squeeze(0))
        Z, A, Z1 = torch.stack(Z), torch.stack(A), torch.stack(Z1)

    print(f"  Training predictor head: {epochs} epochs, frozen encoder, "
          f"cosine loss (same objective as the original JEPA training) ...")
    n = Z.shape[0]
    batch = 256
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        total, steps = 0.0, 0
        for start in range(0, n - batch, batch):
            bi = idx[start:start + batch]
            z_hat = predictor(Z[bi], A[bi])
            loss = (1.0 - F.cosine_similarity(z_hat, Z1[bi], dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1:2d}/{epochs}  loss={total/max(steps,1):.4f}")
    predictor.eval()
    return predictor


def calibrate(jepa, env_id, n=200, seed=7):
    """Identical methodology to walker2d_sensor_actuator.py's v3 fix: real
    rolled-out trajectory, evolving context AND real actions together."""
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
    tau_safe, tau_impasse = mean + 2 * std, mean + 4 * std
    print(f"  Calibration over {n} real steps: mean={mean:.4f} std={std:.4f}")
    print(f"  tau_safe={tau_safe:.4f}  tau_impasse={tau_impasse:.4f}")
    return tau_safe, tau_impasse


if __name__ == "__main__":
    print("=" * 78)
    print("WALKER2D SENSOR/ACTUATOR NODE -- with the ACTUALLY TRAINED encoder")
    print("=" * 78)

    print(f"\nStep 1: load the real trained encoder from {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print(f"  checkpoint's own self-reported stats: {ckpt['stats']}")
    encoder = ProprioceptiveEncoder(ckpt["imu_dim"], ckpt["tactile_dim"], ckpt["embed_dim"])
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    predictor = train_predictor_head(encoder)
    jepa = TrainedWalker2dJEPA(encoder, predictor)

    print("\nStep 2: calibrate the gate from a real rolled-out CPG trajectory.")
    tau_safe, tau_impasse = calibrate(jepa, "Walker2d-v5")
    gate = CalibratedEntropicRouter(tau_safe, tau_impasse)

    print("\nStep 3: run a real episode under the gate -- CPG gait, 300 steps,")
    print("        with one deliberately adversarial torque burst at step 150.")
    env = gym.make("Walker2d-v5")
    rng = np.random.default_rng(42)
    policy = CPGPolicy(rng)
    obs, _ = env.reset(seed=42)
    context = jepa.embed_context(obs)

    ADVERSARIAL_STEP = 150
    SAFE_FALLBACK = np.zeros(ACT_DIM, dtype=np.float32)
    counts = {"COMMIT_TRAJECTORY": 0, "TRIGGER_REPLAN": 0, "STRUCTURAL_IMPASSE": 0}
    adversarial_decision = None
    terminated_for_real, term_step = False, None

    for step in range(300):
        candidate = policy()
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
            obs, _ = env.reset(seed=42 + step)
            context = jepa.embed_context(obs)
            policy.reset()
    env.close()

    print(f"\n{'=' * 78}")
    print("RESULTS -- trained encoder vs. untrained (walker2d_sensor_actuator.py)")
    print("=" * 78)
    print(f"  Decisions over 300 real steps: {counts}")
    print(f"  Adversarial step gate decision: {adversarial_decision.name}")
    print(f"  Real MuJoCo termination occurred: "
          f"{'yes, at step ' + str(term_step) if terminated_for_real else 'no'}")

    assert adversarial_decision != RouteDecision.COMMIT_TRAJECTORY, (
        "FAIL: the trained-encoder gate committed a 100x-out-of-distribution "
        "torque burst to the real actuators."
    )
    print(
        "\nPASS: adversarial burst refused with the trained encoder too "
        f"({adversarial_decision.name}). Compare the COMMIT rate above against "
        "walker2d_sensor_actuator.py's 299/300 -- that comparison, not this "
        "file alone, is the actual test of whether real training changed "
        "anything over the untrained baseline."
    )
