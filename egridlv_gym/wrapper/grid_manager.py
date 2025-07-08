import os
import pandapower as pp
import numpy as np
import pandas as pd


from egridlv_gym.wrapper.grid_utils import (
    min_max_normalize,
)


class GridManager:
    """
    The GridManager class is responsible for managing the grid and associated component
    controllers, internally it uses the pandapower library to simulate the grid.
    Note: this is only a shell class that collects redundant methods, the actual full
    implementation is in the respective grid_manager sub-classes.
    """

    def step(self, actions=None, timestep=0, random_numbers=None):
        """
        Run a simulation step for the grid, applying the given actions and updating the grid state.
        """
        penalties = self.run_controllers(
            actions=actions, timestep=timestep, random_numbers=random_numbers
        )
        aborted = self.run_powerflow(timestep)
        if self.control_q_shielding:
            self.run_shielding_controllers(self.grid)
            aborted = self.run_powerflow(timestep)
        return aborted, penalties

    def reset(self, timestep, np_random):
        """
        Reset the grid state to the initial state at the given timestep.
        """
        self.assign_profiles(timestep)
        self.np_random = np_random

        for storage_controller in self.storage_controllers:
            storage_controller.reset(grid_state=self.grid)
        for ev_controller in self.ev_controllers:
            ev_controller.reset(grid_state=self.grid)
        for hp_controller in self.heat_pump_controllers:
            hp_controller.reset(grid_state=self.grid)
        for pv_controller in self.pv_controllers:
            pv_controller.reset(grid_state=self.grid)

        self.run_powerflow(timestep)

    def run_shielding_controllers(self, grid_state):
        """
        Runs only the controllers supporting the VDE piecewise linear control scheme.
        """
        for bes in self.storage_controllers:
            bes.vde_adjust(self.grid, "voltage_reactive")
        for pv in self.pv_controllers:
            pv.voltage_reactive_q_control(grid_state)

    def run_controllers(self, actions, timestep, random_numbers):
        """
        Run the controllers for each component in the grid.
        Returns the penalties incurred by each controller type (sum of all controllers of that type).
        """

        storage_offset = 0
        ev_offset = self.storages_count
        hp_offset = self.storages_count + self.ev_count
        pv_offset = self.storages_count

        if self.control_p:
            p_actions = actions[: self.p_controllers_count]
        else:
            p_actions = [0] * (self.storages_count + self.ev_count + self.hp_count)
        if self.control_q:
            q_actions = actions[self.p_controllers_count :]
        else:
            q_actions = [0] * (self.storages_count + self.pv_count)

        bes_penalties = 0
        ev_penalties = 0
        pv_penalties = 0
        hp_penalties = 0

        for index, storage_controller in enumerate(self.storage_controllers):
            bes_penalty = storage_controller.run(
                grid_state=self.grid,
                p_action=p_actions[index + storage_offset],
                q_action=q_actions[index + storage_offset],
                timestep=timestep,
            )
            bes_penalties += bes_penalty

        for index, ev_controller in enumerate(self.ev_controllers):
            ev_penalty = ev_controller.run(
                grid_state=self.grid,
                p_action=p_actions[index + ev_offset],
                timestep=timestep,
                random_numbers=random_numbers[index],
            )
            ev_penalties += ev_penalty

        for index, hp_controller in enumerate(self.heat_pump_controllers):
            hp_penalty = hp_controller.run(
                grid_state=self.grid,
                p_action=p_actions[index + hp_offset],
                timestep=timestep,
            )
            hp_penalties += hp_penalty

        for index, pv_controller in enumerate(self.pv_controllers):
            pv_penalty = pv_controller.run(
                grid_state=self.grid,
                q_action=q_actions[index + pv_offset],
                timestep=timestep,
            )
            pv_penalties += pv_penalty

        return {
            "battery_penalty": bes_penalties,
            "charger_penalty": ev_penalties,
            "pv_penalty": pv_penalties,
            "hp_penalty": hp_penalties,
        }

    def run_powerflow(self, t):
        """
        Run a power flow calculation for the grid.
        Returns True if the power flow calculation failed to converge.
        """
        try:
            pp.runpp(
                self.grid,
                init="results",
                calculate_voltage_angles=False,
                tolerance_mva=1e-6,
            )
        except:
            print("Failed to converge at time step: ", t)
            print(pp.diagnostic(self.grid))
            return True
        return False

    def get_controllers_count(self):
        """
        Returns a tuple with the number of controllers for each component type.
        """
        return self.storages_count, self.pv_count, self.ev_count, self.hp_count

    def get_control_space(self):
        """
        Returns the dimensions of the action space for the agent.
        """
        return self.p_controllers_count, self.q_controllers_count

    def get_current_power(self):
        """
        Returns the current power consumption and generation for each component type.
        """
        loads = self.grid.load.loc[self.pure_loads, "p_mw"].values
        sgen = self.grid.sgen.loc[:, "p_mw"].values
        storage = np.array(
            [sc.p_mw for sc in self.storage_controllers], dtype=np.float64
        )
        ev_p = np.array([evc.p_mw for evc in self.ev_controllers], dtype=np.float64)
        hp_p = np.array(
            [hpc.p_mw for hpc in self.heat_pump_controllers], dtype=np.float64
        )

        return loads, sgen, storage, ev_p, hp_p

    def get_storage_soc(self):
        return np.array([sc.soc_percent for sc in self.storage_controllers])

    def get_ev_soc(self):
        return np.array([ev.soc_percent for ev in self.ev_controllers])

    def get_ev_present(self):
        return np.array([ev.at_home for ev in self.ev_controllers])

    def get_heat_storage(self):
        return np.array([hpc.storage() for hpc in self.heat_pump_controllers])

    def get_prev_heat_pump_power(self):
        return np.array(
            [hpc.get_prev_load() for hpc in self.heat_pump_controllers],
            dtype=np.float64,
        )

    def get_line_loading(self):
        return self.grid.res_line.loading_percent.values

    def get_line_losses(self):
        return self.grid.res_line.pl_mw.values

    def get_transformer_losses(self):
        return self.grid.res_trafo.pl_mw.values

    def get_transformer_loading(self):
        return self.grid.res_trafo.loading_percent.values

    def get_ext_power(self):
        return self.grid.res_ext_grid.p_mw.values

    def get_voltage_pu(self, normalize=False):
        voltages = self.grid.res_bus.vm_pu.values
        # Remove the bus connected to the external grid
        voltages = voltages[1:]
        if normalize:
            min_max_normalizer = np.vectorize(min_max_normalize)
            voltages = min_max_normalizer(voltages, 0.75, 1.25)
        return voltages

    def get_pv_reactive_power(self):
        return self.grid.sgen.loc[:, "q_mvar"].values

    def get_pv_curtailment(self):
        return np.array([pv.get_curtailment() for pv in self.pv_controllers])

    def get_max_net_power(self):
        """
        The theoretical maximum power that can be generated by the grid

        """
        return self.max_sgen + self.max_storage_p + self.max_ev_p - self.min_load_p

    def get_min_net_power(self):
        """
        The theoretical maximum power that can be consumed by the grid
        """
        return self.min_sgen - self.max_storage_p - self.max_load_p - self.max_ev_p

    def get_controllable_storage_power(self):
        """
        Returns the maximum power that can be controlled by the storage units.
        """
        max_p = [storage.max_p_mw for storage in self.storage_controllers]
        max_e = [storage.max_e_mwh for storage in self.storage_controllers]
        return (max_p, max_e)

    def get_controllable_ev_power(self):
        """
        Returns the maximum power that can be controlled by the EV chargers.
        """
        max_p = [ev.max_p_mw for ev in self.ev_controllers]
        max_e = [ev.max_e_mwh for ev in self.ev_controllers]
        return max_p, max_e

    def get_controllable_hp_power(self):
        """
        Returns the maximum power that can be controlled by the heat pumps.
        """
        return [hp.max_p_mw for hp in self.heat_pump_controllers]

    def load_car_distributions(self):
        base_path = os.path.join(os.path.dirname(__file__), "../../data")
        file_path = os.path.join(base_path, "car_distributions.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

        data = pd.read_csv(file_path)
        self.ev_arrival_probabilities = data["arrival"]
        self.ev_departure_probabilities = data["leave"]
