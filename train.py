"""
Individual RL mini-project: single HVAC zone, single Q-learning agent.
Zone modeled as an office space with solar heat gain + daytime occupancy.
Change the parameters in ZONE below to match your actually-assigned room
if it differs (e.g. remove solar_gain_amp for a server room, raise
internal_load for an equipment room, etc.)
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import HVACZoneEnv
from agent import QLearningAgent

STATE_BINS = (13, 8, 8)
N_ACTIONS = 3
N_EPISODES = 400
SEED = 42

ZONE = HVACZoneEnv(
    "office_zone",
    target_temp=23.0,
    tolerance=1.0,
    internal_load=1.0,
    solar_gain_amp=6.0,
    occupancy_profile=lambda h: 2.0 if 9 <= h <= 18 else 0.0,
    seed=SEED,
)

agent = QLearningAgent(N_ACTIONS, STATE_BINS, seed=SEED)
episode_rewards, success_rates = [], []

for ep in range(N_EPISODES):
    state = ZONE.reset()
    total_reward, in_tolerance, steps, done = 0.0, 0, 0, False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = ZONE.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if abs(info["temp"] - ZONE.target_temp) <= ZONE.tolerance:
            in_tolerance += 1
        steps += 1
    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    success_rates.append(in_tolerance / steps * 100)

print(f"Final avg reward (last 20 eps): {np.mean(episode_rewards[-20:]):.2f}")
print(f"Final comfort success rate:     {np.mean(success_rates[-20:]):.1f}%")
print(f"Final epsilon:                  {agent.epsilon:.3f}")

# ---- Learning curve ----
rewards = np.array(episode_rewards)
smoothed = np.convolve(rewards, np.ones(15) / 15, mode="valid")
plt.figure(figsize=(8, 5))
plt.plot(rewards, alpha=0.3, label="Episode reward")
plt.plot(range(14, 14 + len(smoothed)), smoothed, linewidth=2, label="15-ep moving avg")
plt.title("Learning Curve - Office Zone HVAC Agent")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve1.png", dpi=150)
plt.close()

# ---- Comfort success rate ----
succ = np.array(success_rates)
smoothed_s = np.convolve(succ, np.ones(15) / 15, mode="valid")
plt.figure(figsize=(8, 5))
plt.plot(succ, alpha=0.3, label="Success rate (%)")
plt.plot(range(14, 14 + len(smoothed_s)), smoothed_s, linewidth=2, label="15-ep moving avg")
plt.title("Comfort Success Rate - Office Zone HVAC Agent")
plt.xlabel("Episode")
plt.ylabel("% time within comfort band")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig("success_rate1.png", dpi=150)
plt.close()

# ---- Epsilon decay curve (for CO5 exploration-exploitation evidence) ----
eps_curve = [1.0]
e = 1.0
for _ in range(N_EPISODES - 1):
    e = max(0.05, e * 0.995)
    eps_curve.append(e)
plt.figure(figsize=(8, 5))
plt.plot(eps_curve, color="darkorange", linewidth=2)
plt.title("Epsilon Decay - Exploration to Exploitation")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.tight_layout()
plt.savefig("epsilon_decay1.png", dpi=150)
plt.close()

with open("trained_agent1.pkl", "wb") as f:
    pickle.dump(agent.Q, f)

print("Saved: learning_curve.png, success_rate.png, epsilon_decay.png, trained_agent.pkl")
