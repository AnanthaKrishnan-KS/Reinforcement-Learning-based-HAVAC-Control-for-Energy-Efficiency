import random
import numpy as np


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(self, n_actions, state_bins, alpha=0.15, gamma=0.95,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=None):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros(state_bins + (n_actions,))
        self.rng = random.Random(seed)

    def select_action(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.Q[next_state])
        target = reward + (0 if done else self.gamma * best_next)
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
