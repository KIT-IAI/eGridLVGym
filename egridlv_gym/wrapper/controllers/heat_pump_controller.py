import numpy as np


class HeatPumpController:
    def __init__(
        self,
        load,
        id,
        window_size=24,
        power_scaling=1,
        control_p=True,
    ):
        super().__init__()
        self.load = load
        self.id = id
        self.error = np.zeros(window_size)
        self.max_p_mw = load.sn_mva * power_scaling
        self.prev_load = 0
        self.max_error = self.max_p_mw

        self.control_p = control_p

        self.p_mw = 0

    def run(self, grid_state, p_action, timestep):
        penalty = 0
        self.prev_load = grid_state.load.at[self.id, "p_mw"]
        if self.control_p:
            penalty = self._p_action(grid_state, p_action)
        return penalty

    def reset(self, grid_state):
        self.error = np.zeros(len(self.error))

    def _p_action(self, grid_state, p_action):
        scaled_action = self.scale_action(p_action)
        self.p_mw = scaled_action * self.prev_load
        self.p_mw = np.clip(self.p_mw, 0, self.max_p_mw)
        grid_state.load.at[self.id, "p_mw"] = self.p_mw
        self.error = np.roll(self.error, 1)
        self.error[0] = self.p_mw - self.prev_load
        penalty = self.calculate_penalty()
        return penalty

    def calculate_penalty(self):
        error_sum = np.abs(np.sum(self.error))
        if error_sum > self.max_error:
            return error_sum - self.max_error
        return 0

    def get_prev_load(self):
        return self.prev_load

    def storage(self):
        return np.sum(self.error)

    def scale_action(self, unscaled, to_min=0.5, to_max=1.5, from_min=-1, from_max=1):
        return (to_max - to_min) * (unscaled - from_min) / (
            from_max - from_min
        ) + to_min
