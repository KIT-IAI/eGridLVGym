import numpy as np


class EVChargerController:
    def __init__(
        self,
        storage,
        id,
        arrival_probabilities,
        departure_probabilities,
        soc_percent_charge_limit=0.9,
        soc_percent_discharge_limit=0.1,
        control_p=True,
        np_random=None,
        power_scaling=1,
    ):
        self.id = id
        self.storage = storage
        self.bus = storage.bus
        self.soc_percent = storage.soc_percent
        self.soc_percent_charge_limit = soc_percent_charge_limit
        self.soc_percent_discharge_limit = soc_percent_discharge_limit
        self.min_p_mw = storage.min_p_mw * power_scaling
        self.max_p_mw = storage.max_p_mw * power_scaling
        self.max_e_mwh = storage.max_e_mwh * power_scaling
        self.sn_mva = storage.sn_mva * power_scaling
        self.p_mw = 0
        self.p_control = control_p
        self.np_random = np_random

        self.arrival_probabilities = arrival_probabilities
        self.departure_probabilities = departure_probabilities

    def run(self, grid_state, timestep, p_action, random_numbers):
        penalty = 0
        if self.at_home:
            if self.p_control:
                penalty = self.run_at_home(grid_state, timestep, p_action)
            else:
                self.rule_based_p_control(grid_state, timestep)
            self.update_soc_from_previous(grid_state)
        else:
            grid_state.storage.at[self.id, "p_mw"] = 0
            self.soc_percent = 0
            grid_state.storage.at[self.id, "soc_percent"] = 0
        self.update_state(timestep, grid_state, random_numbers)
        return penalty

    def reset(self, grid_state):
        self.soc_percent = 0.5
        grid_state.storage.at[self.id, "soc_percent"] = 0.5
        self.at_home = True
        grid_state.storage.at[self.id, "in_service"] = True

    def run_at_home(self, grid_state, timestep, p_action):
        if self.p_control:
            penalty = self.p_action(grid_state, p_action)
        return penalty

    def p_action(self, net, action):
        """
        Pandapower storage model:
        positive power -> charge
        negative power -> discharge
        """
        penalty = 0
        p_mw = action * self.max_p_mw
        e_current = self.soc_percent * self.max_e_mwh
        if (p_mw > ((self.max_e_mwh - e_current) * 4)) or (p_mw > self.max_p_mw):
            p_mw = (self.max_e_mwh - e_current) * 4
            p_mw = min(p_mw, self.max_p_mw)
            penalty = 1
        elif (p_mw < -(e_current * 4)) or (p_mw < -self.max_p_mw):
            p_mw = -e_current * 4
            p_mw = max(p_mw, -self.max_p_mw)
            penalty = 1

        self.p_mw = p_mw
        net.storage.at[self.id, "p_mw"] = p_mw
        return penalty

    def update_state(self, timestep, grid_state, random_numbers):
        """
        Generates Departure/Arrival events based on the time of day.
        This is based on two distributions for the arrival and departure times.
        The next event is drawn from the respective distribution and the state is updated accordingly.
        """
        departure_number = random_numbers[0]
        soc_random_bracket = random_numbers[1]
        soc_random_value = random_numbers[2]
        timestep = timestep % 96
        if self.at_home:
            departure_prob = self.departure_probabilities[timestep]
            if departure_number <= departure_prob:
                self.at_home = False
                self.soc_percent = 0
                grid_state.storage.at[self.id, "soc_percent"] = 0
                grid_state.storage.at[self.id, "in_service"] = False
        else:
            arrival_prob = self.arrival_probabilities[timestep]
            if departure_number <= arrival_prob:
                self.at_home = True
                grid_state.storage.at[self.id, "in_service"] = True
                if soc_random_bracket <= 0.5:
                    self.soc_percent = 0.3 + soc_random_value * 0.2
                elif soc_random_bracket <= 0.8:
                    self.soc_percent = 0.5 + soc_random_value * 0.3
                else:
                    self.soc_percent = 0.8 + soc_random_value * 0.2
                grid_state.storage.at[self.id, "soc_percent"] = self.soc_percent

    def update_soc_from_previous(self, grid):
        """Updates the state of charge based on the previous time step."""
        delta_energy = self.p_mw
        self.current_energy_change = delta_energy * (1 / 4)
        # Update the state of charge (SoC) based on the delta energy and
        # the maximum stored energy
        self.soc_percent += self.current_energy_change / self.max_e_mwh
        # Ensure soc_percent does not exceed 100%
        self.soc_percent = min(self.soc_percent, 1)
        self.soc_percent = max(self.soc_percent, 0)
        grid.storage.at[self.id, "soc_percent"] = self.soc_percent

    def rule_based_p_control(self, grid, timestep):
        """Rule-based control for the EV charger."""
        e_current = self.soc_percent * self.max_e_mwh
        space = self.max_e_mwh - e_current
        if self.soc_percent < 1:
            self.p_mw = min(self.max_p_mw, space * 4)
        else:
            self.p_mw = 0
        grid.storage.at[self.id, "p_mw"] = self.p_mw
