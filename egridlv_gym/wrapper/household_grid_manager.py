import numpy as np
import pandapower as pp
import simbench as sb

# import pandas as pd
# import pickle

from egridlv_gym.wrapper.controllers.storage_controller import StorageController
from egridlv_gym.wrapper.controllers.ev_charger_controller import EVChargerController
from egridlv_gym.wrapper.controllers.heat_pump_controller import HeatPumpController
from egridlv_gym.wrapper.controllers.pv_controller import PVController
from egridlv_gym.wrapper.grid_manager import GridManager


from egridlv_gym.wrapper.grid_utils import (
    get_simbench_grid,
    min_max_normalize,
)


class HouseholdGridManager(GridManager):
    def __init__(
        self,
        power_scaling=1,
        control_p=True,
        control_q=False,
        control_q_shielding=False,
        np_random=None,
    ) -> None:
        self.control_q_shielding = control_q_shielding
        self.reference_grid = get_simbench_grid()
        self.create_load_profiles()
        self.load_car_distributions()

        self.power_scaling = power_scaling
        self.control_p = control_p
        self.control_q = control_q

        # Indices for the reference grid
        self.load_index = 1
        self.sgen_index = 0
        self.storage_index = 1
        self.hp_index = 25

        grid = pp.create_empty_network()
        b1 = pp.create_bus(grid, name="Bus 1", vn_kv=20.0)
        b2 = pp.create_bus(grid, name="Bus 2", vn_kv=0.4)
        b3 = pp.create_bus(grid, name="Bus 3", vn_kv=0.4)

        pp.create_line(
            grid,
            b2,
            b3,
            length_km=0.1,
            std_type="NA2XS2Y 1x240 RM/25 12/20 kV",
            name="line_house",
        )

        pp.create_ext_grid(grid, b1, name="Grid Connection")
        pp.create_transformer(grid, b1, b2, std_type="0.4 MVA 20/0.4 kV")

        household_load = self.reference_grid.load.loc[self.load_index]
        heatpump_load = self.reference_grid.load.loc[self.hp_index]
        storage = self.reference_grid.storage.loc[self.storage_index]
        sgen = self.reference_grid.sgen.loc[self.sgen_index]

        pure_load = pp.create_load(
            grid,
            bus=b3,
            p_mw=household_load["p_mw"],
            q_mvar=household_load["q_mvar"],
            sn_mva=household_load["sn_mva"] * self.power_scaling,
            const_z_percent=0.0,
            const_i_percent=0.0,
            name="Household Load",
        )

        self.pure_loads = [pure_load]

        self.heatpump_sid = pp.create_load(
            grid,
            bus=b3,
            p_mw=heatpump_load["p_mw"],
            q_mvar=heatpump_load["q_mvar"],
            sn_mva=heatpump_load["sn_mva"] * self.power_scaling,
            const_z_percent=0.0,
            const_i_percent=0.0,
            name="Heatpump Load",
        )

        self.storage_sid = pp.create_storage(
            grid,
            bus=b3,
            p_mw=0,
            q_mvar=0,
            max_q_mvar=storage["max_q_mvar"],
            min_q_mvar=storage["min_q_mvar"],
            sn_mva=storage["sn_mva"] * self.power_scaling,
            soc_percent=0.5,
            min_e_mwh=storage["min_e_mwh"],
            max_e_mwh=storage["max_e_mwh"],
            max_p_mw=storage["sn_mva"],
            min_p_mw=storage["min_p_mw"],
            scaling=self.power_scaling,
            in_service=True,
            efficiency_percent=storage["efficiency_percent"],
            name="Storage Bus 3",
        )

        self.sgen_id = pp.create_sgen(
            grid,
            bus=b3,
            p_mw=sgen["p_mw"],
            q_mvar=sgen["q_mvar"],
            sn_mva=sgen["sn_mva"] * self.power_scaling,
            min_p_mw=sgen["min_p_mw"],
            max_p_mw=sgen["max_p_mw"],
        )

        self.ev_sid = pp.create_storage(
            grid,
            bus=b3,
            p_mw=0,
            q_mvar=0,
            sn_mva=0.008 * self.power_scaling,
            soc_percent=0.5,
            min_e_mwh=0,
            max_e_mwh=0.075,
            max_p_mw=0.008,
            min_p_mw=0,
            scaling=self.power_scaling,
            in_service=True,
            efficiency_percent=0.95,
            name="EV Charger Bus 3",
        )

        pv_controller = PVController(
            id=self.sgen_index,
            pv=grid.sgen.loc[0],
            power_scaling=self.power_scaling,
        )

        storage_controller = StorageController(
            id=self.storage_sid,
            storage=grid.storage.loc[0],
            control_p=control_p,
            control_q=control_q,
            control_q_shielding=control_q_shielding,
        )

        ev_controller = EVChargerController(
            id=self.ev_sid,
            storage=grid.storage.loc[1],
            arrival_probabilities=self.ev_arrival_probabilities,
            departure_probabilities=self.ev_departure_probabilities,
            control_p=control_p,
            np_random=np_random,
        )

        heatpump_controller = HeatPumpController(
            id=self.heatpump_sid,
            load=heatpump_load,
            window_size=96,
        )

        self.storage_controllers = [storage_controller]
        self.ev_controllers = [ev_controller]
        self.pv_controllers = [pv_controller]
        self.heat_pump_controllers = [heatpump_controller]

        self.storages_count = 1
        self.pv_count = 1
        self.ev_count = 1
        self.hp_count = 1

        if self.control_p:
            self.p_controllers_count = (
                self.ev_count + self.storages_count + self.hp_count
            )
        else:
            self.p_controllers_count = 0
        if self.control_q:
            self.q_controllers_count = self.pv_count + self.p_controllers_count
        else:
            self.q_controllers_count = 0

        self.grid = grid
        # Replacing the existing 160-kVA transformer with a 250-kVA Siemens oil-immersed distribution transformer
        # self.grid.trafo.loc[0, "sn_mva"] = 0.25
        self.grid.storage.at[0, "soc_percent"] = 0.5

        self.calculate_boundary_values()

    def create_load_profiles(self):
        profiles = sb.get_absolute_values(
            self.reference_grid, profiles_instead_of_study_cases=True
        )
        self.active_profiles = profiles[("load", "p_mw")]
        self.reactive_profiles = profiles[("load", "q_mvar")]
        self.sgen_profiles = profiles[("sgen", "p_mw")]

    def calculate_boundary_values(self):

        self.min_load_p = self.active_profiles[self.load_index].min()
        self.max_load_p = self.active_profiles[self.load_index].max()

        self.max_hp_p = self.active_profiles[self.hp_index].max()

        self.min_load_q = self.reactive_profiles[self.load_index].min()
        self.max_load_q = self.reactive_profiles[self.load_index].max()

        self.min_sgen = self.sgen_profiles[0].min()
        self.max_sgen = self.sgen_profiles[0].max()

        self.max_storage_p = self.reference_grid.storage.loc[0]["sn_mva"]
        self.max_ev_p = self.reference_grid.storage.loc[1]["sn_mva"]
        self.max_e_mwh = self.reference_grid.storage.loc[0]["max_e_mwh"]

    def assign_profiles(self, timestep):
        self.grid.load.loc[0, "p_mw"] = (
            self.active_profiles[self.load_index][timestep] * self.power_scaling
        )
        self.grid.load.loc[0, "q_mw"] = (
            self.reactive_profiles[self.load_index][timestep] * self.power_scaling
        )
        self.grid.load.loc[1, "p_mw"] = (
            self.active_profiles[self.hp_index][timestep] * self.power_scaling
        )
        self.grid.load.loc[1, "q_mw"] = (
            self.reactive_profiles[self.hp_index][timestep] * self.power_scaling
        )
        self.grid.sgen.loc[0, "p_mw"] = (
            self.sgen_profiles[self.sgen_index][timestep] * self.power_scaling
        )

    def get_uncontrolled_power(self, horizon, timestep, normalize=False):
        """
        Get the uncontrolled power consumption and generation for the next horizon
        """
        load = self.active_profiles[self.load_index][timestep : timestep + horizon]
        sgen = self.sgen_profiles[self.sgen_index][timestep : timestep + horizon]
        hp_p = self.active_profiles[self.hp_index][timestep : timestep + horizon]

        min_max_normalizer = np.vectorize(min_max_normalize)
        if normalize:
            load = min_max_normalizer(load, 0, self.max_sgen)
            sgen = min_max_normalizer(sgen, 0, self.max_sgen)
            hp_p = min_max_normalizer(hp_p, 0, self.max_hp_p)

        load = np.array(load, dtype=np.float32)
        load = load[:, np.newaxis]
        sgen = np.array(sgen, dtype=np.float32)
        sgen = sgen[:, np.newaxis]
        hp_p = np.array(hp_p, dtype=np.float32)
        hp_p = hp_p[:, np.newaxis]

        return load, sgen, hp_p

    def get_controlled_power(
        self,
    ):
        """
        Get the controlled power consumption and generation for the current timestep
        """
        storage = [self.grid.storage.p_mw.at[0]]
        ev_power = [self.grid.storage.p_mw.at[1]]
        heatpump = [self.grid.load.p_mw.at[1]]

        return storage, ev_power, heatpump

    def get_observation_space(self):
        return 1, 1, 1, 1, 1

    def get_control_space(self):
        """
        Returns the maximum number of actions that can be taken by the agents (P and Q)
        """
        return 3, 3
