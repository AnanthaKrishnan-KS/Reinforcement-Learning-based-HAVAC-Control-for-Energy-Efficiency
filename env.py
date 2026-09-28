import numpy as np


class HVACZoneEnv:
    """
    Simplified thermal simulation of a single HVAC-controlled zone.
    Different constructor parameters give each zone a distinct thermal
    personality (constant internal load, solar gain, occupancy pattern, etc.)
    so that 4 teammates can each train a genuinely different agent.
    """

    def __init__(self, zone_name, target_temp=22.0, tolerance=1.0,
                 alpha=0.06,                 # heat exchange rate with outside air
                 heat_effect=1.4, cool_effect=-1.4,
                 internal_load=0.0,          # constant internal heat gain (e.g. servers)
                 solar_gain_amp=0.0,         # peak solar heat gain (glass walls)
                 occupancy_profile=None,     # function(hour) -> heat load
                 energy_cost=1.0,
                 episode_length=96,          # 96 steps/day at 15-min resolution
                 seed=None):
        self.zone_name = zone_name
        self.target_temp = target_temp
        self.tolerance = tolerance
        self.alpha = alpha
        self.heat_effect = heat_effect
        self.cool_effect = cool_effect
        self.internal_load = internal_load
        self.solar_gain_amp = solar_gain_amp
        self.occupancy_profile = occupancy_profile or (lambda h: 0.0)
        self.energy_cost = energy_cost
        self.episode_length = episode_length
        self.actions = [0, 1, 2]  # 0 = off, 1 = heat, 2 = cool
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _outside_temp(self, t):
        hour = (t / self.episode_length) * 24
        # daily sinusoidal outside temperature, roughly 15C to 33C
        return 24 + 9 * np.sin((hour - 9) / 24 * 2 * np.pi)

    def reset(self):
        self.t = 0
        self.temp = self.target_temp + self.rng.uniform(-3, 3)
        return self._get_state()

    def _get_state(self):
        hour = int((self.t / self.episode_length) * 24) % 24
        outside = self._outside_temp(self.t)
        temp_err = self.temp - self.target_temp
        temp_bin = int(np.clip(round(temp_err), -6, 6)) + 6          # 0..12
        outside_bin = int(np.clip(round((outside - 15) / 3), 0, 7))  # 0..7
        hour_bin = hour // 3                                          # 0..7
        return (temp_bin, outside_bin, hour_bin)

    def step(self, action):
        hour = (self.t / self.episode_length) * 24
        outside = self._outside_temp(self.t)
        solar = 0.0
        if 6 <= hour <= 18:
            solar = self.solar_gain_amp * max(0.0, np.sin((hour - 6) / 12 * np.pi))
        occ_load = self.occupancy_profile(hour)

        effect, energy_used = 0.0, 0.0
        if action == 1:
            effect, energy_used = self.heat_effect, 1.0
        elif action == 2:
            effect, energy_used = self.cool_effect, 1.0

        drift = self.alpha * (outside - self.temp)
        self.temp += (drift + effect + 0.05 * (self.internal_load + solar + occ_load)
                      + self.rng.normal(0, 0.15))

        temp_err = abs(self.temp - self.target_temp)
        comfort_penalty = temp_err ** 2
        reward = -(comfort_penalty + self.energy_cost * energy_used * 0.5)

        self.t += 1
        done = self.t >= self.episode_length
        info = {"temp": self.temp, "outside": outside, "energy": energy_used}
        return self._get_state(), reward, done, info
