"""
Live-demo script (CO3 requirement): runs the trained agent with its
greedy (learned) policy for several steps and prints the full
state -> action -> reward -> next_state cycle for the viva demo.
"""
import pickle
from env import HVACZoneEnv
from agent import QLearningAgent

ACTION_NAMES = {0: "OFF", 1: "HEAT", 2: "COOL"}

with open("trained_agent1.pkl", "rb") as f:
    Q = pickle.load(f)

env = HVACZoneEnv("office_zone", target_temp=23.0, internal_load=1.0,
                   solar_gain_amp=6.0,
                   occupancy_profile=lambda h: 2.0 if 9 <= h <= 18 else 0.0,
                   seed=7)
agent = QLearningAgent(3, (13, 8, 8))
agent.Q = Q

state = env.reset()
print("DEMO: Office Zone HVAC Agent | trained greedy policy\n")
print(f"{'Step':<5}{'Outside(C)':<12}{'State':<16}{'Action':<8}{'Reward':<10}{'NextTemp':<10}")
print("-" * 65)

for step in range(10):
    action = agent.select_action(state, greedy=True)
    next_state, reward, done, info = env.step(action)
    print(f"{step:<5}{info['outside']:<12.2f}{str(state):<16}{ACTION_NAMES[action]:<8}"
          f"{reward:<10.2f}{info['temp']:<10.2f}")
    state = next_state
    if done:
        break

print(f"\nTarget: {env.target_temp} C | Tolerance: +/-{env.tolerance} C")
