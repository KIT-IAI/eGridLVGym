import numpy as np



class StorageController:
    def __init__(
        self,
        storage,
        id,
        soc_percent_charge_limit=0.90,
        soc_percent_discharge_limit=0.10,
        efficiency_percent=0.9,
        q_mode="underexcited",
        control_p=True,
        control_q=False,
        control_q_shielding=False,
        power_scaling=1,
    ):
        self.sid = id
        self.bus = storage.bus
        self.soc_percent = storage.soc_percent
        self.soc_percent_charge_limit = soc_percent_charge_limit
        self.soc_percent_discharge_limit = soc_percent_discharge_limit
        self.min_p_mw = -storage.sn_mva * power_scaling
        self.max_p_mw = storage.sn_mva * power_scaling
        self.max_e_mwh = storage.max_e_mwh * power_scaling
        self.p_mw = 0
        self.efficiency_percent = storage.efficiency_percent
        # self-discharge per day as a share of the stored energy: 0.001/day = 3 %/month, upper end of Li-ion
        # battery storage incl. BMS consumption (assumption)
        self.discharge_loss = 0.001
        self.q_mode = q_mode
        self.sn_mva = storage.sn_mva * power_scaling
        self.c_precision = 4
        self.p_control = control_p
        self.q_control = control_q
        self.control_q_shielding = control_q_shielding
        self.bus = storage.bus
        self.power_scaling = power_scaling

        self.v_1 = 0.93
        self.v_2 = 0.97
        self.v_4 = 1.03
        self.v_5 = 1.07

        # Nominal reactive power value
        self.nomi_q_mvar = 0

        if self.sn_mva > 0.0046:
            self.min_cos_phi = 0.9
            self.reactive_power_performance_limit = 0.200
        else:
            self.min_cos_phi = 0.95
            self.reactive_power_performance_limit = 0.200

        self.available_reactive_power = self.calc_allowed_reactive_power(
            self.min_cos_phi
        )
        self.norm_min_q_mvar = -self.available_reactive_power
        self.norm_max_q_mvar = self.available_reactive_power
        self.min_q_mvar = self.norm_min_q_mvar * self.sn_mva
        self.max_q_mvar = self.norm_max_q_mvar * self.sn_mva

        self.q_mvar = 0
        self.p_mw = 0
        self.q_mode = "overexcited"

    def run(self, grid_state, timestep, p_action=None, q_action=None):
        p_penalty = 0
        if self.p_control:
            p_penalty = self.p_action(grid_state, p_action)
        else:
            self.rule_based_p_control(grid_state)

        grid_state.storage.at[self.sid, "p_mw"] = self.p_mw

        if self.q_control:
            self.q_action(grid_state, q_action)
            grid_state.storage.at[self.sid, "q_mvar"] = self.q_mvar
        self.update_soc_from_previous(grid_state)

        grid_state.storage.at[self.sid, "p_mw"] = self.p_mw

        return p_penalty

    def reset(self, grid_state):
        self.soc_percent = 0.5
        grid_state.storage.at[self.sid, "soc_percent"] = self.soc_percent
        self.p_mw = 0
        self.q_mvar = 0
        grid_state.storage.at[self.sid, "p_mw"] = 0
        grid_state.storage.at[self.sid, "q_mvar"] = 0
        self.q_mode = "overexcited"

    def p_action(self, grid, action):
        """
        Pandapower storage model:
        positive power -> charge
        negative power -> discharge
        """
        penalty = 0
        p_mw = action * self.max_p_mw
        e_current = self.soc_percent * self.max_e_mwh
        # one-step energy headroom incl. efficiency (see update_soc_from_previous):
        # charging stores P * eta * dt, discharging removes P / eta * dt
        p_charge_max = (self.max_e_mwh - e_current) * 4 / self.efficiency_percent
        p_discharge_max = e_current * 4 * self.efficiency_percent
        if (p_mw > p_charge_max) or (p_mw > self.max_p_mw):
            p_mw = min(p_charge_max, self.max_p_mw)
            penalty = 1
        elif (p_mw < -p_discharge_max) or (p_mw < -self.max_p_mw):
            p_mw = max(-p_discharge_max, -self.max_p_mw)
            penalty = 1
        p_mw = round(p_mw, self.c_precision)
        self.p_mw = p_mw
        return penalty

    def q_action(self, grid_state, action):
        cos_range = 1 - self.min_cos_phi
        cos_phi_action = action * cos_range
        if cos_phi_action > 0:
            self.q_mode = "overexcited"
            cos_phi = 1 - cos_phi_action
        else:
            self.q_mode = "underexcited"
            cos_phi = 1 + cos_phi_action
        self.constant_power_factor_control(grid_state, cos_phi)

    def vde_adjust(self, grid_state, mode):
        if mode == "voltage_reactive":
            self.voltage_reactive_q_control(grid_state)
        elif mode == "const_power_factor":
            self.constant_power_factor_control(grid_state, 0.9)
        grid_state.storage.at[self.sid, "p_mw"] = self.p_mw
        grid_state.storage.at[self.sid, "q_mvar"] = self.q_mvar
        pass

    def update_soc_from_previous(self, grid):
        """Updates the state of charge based on the previous time step."""
        # charging stores P * eta * dt, discharging removes P / eta * dt
        if self.p_mw >= 0:
            delta_energy = self.p_mw * self.efficiency_percent / 4
        else:
            delta_energy = self.p_mw / self.efficiency_percent / 4
        # Loss of energy due to self-discharge
        discharge_loss = (self.discharge_loss / 96) * self.soc_percent * self.max_e_mwh
        self.current_energy_change = delta_energy - discharge_loss
        # Update the state of charge (SoC) based on the delta energy and
        # the maximum stored energy
        self.soc_percent += self.current_energy_change / self.max_e_mwh
        # Ensure soc_percent does not exceed 100%
        self.soc_percent = min(self.soc_percent, 1)
        self.soc_percent = max(self.soc_percent, 0)
        grid.storage.at[self.sid, "soc_percent"] = self.soc_percent

    def calc_allowed_reactive_power(self, cos_phi_value):
        """Calculate the available reactive power (q_available) for a given power factor (cos phi).
        This function calculates the phase angle (phi) and the tangent of
        the phase angle (tan phi) based on the given power factor (cos phi).
        It then multiplies the tangent of the phase angle by the power
        factor to obtain the Qvb value, which represents the available
        reactive power in an electrical system, typically related to
        the apparent power (S).
        Args:
            cos_phi_value (float): The power factor (cos phi) value.
        Returns:
            float: The calculated q_available value.
        Notes:
            q_available_095 = self.calc_allowed_reactive_power(0.95)
            print("Qvb for cos phi = 0.95:", q_available_095)
        """
        # Calculate the phase angle phi
        phi = np.arccos(cos_phi_value)
        # Calculate the tangent of the phase angle
        tan_phi = np.tan(phi)
        # Multiply the tangent of the phase angle by the power factor
        q_available = tan_phi * cos_phi_value
        return round(q_available, self.c_precision)

    def voltage_reactive_q_control(self, grid):
        """(1): Q(V) - Voltage Reactive Power Control Mode
        Implements voltage-reactive power control (Q(V) characteristic) for PV systems.
        This method controls reactive power (Q) in response to the grid voltage (V),
        as described by the Q(V) curve in accordance with VDE-AR-N 4105:2026-03
        guidelines.
        It ensures that power outputs (real and reactive) are within the technical
        operational limits of the inverter. Depending on the normalized momentary real
        power, it either restricts the reactive power to 10% of the maximum real power
        as per VDE guidelines, or operates in a voltage-reactive power mode where Q is
        determined based on the voltage level at the grid connection point.
        The function also adjusts the real power output to respect the total apparent
        power limit, while saving unbounded reactive power values for potential
        future usage or analysis.
        Low Power Range Condition:
            For the range 0 <= Pmom/PEmax < 0.2, the reactive power should not exceed
            10%/20% of the maximum active power.
        Args:
            grid --  egrid
            mom_p_mw -- momentary active power desired by run_battery_p_control
        Returns:
            self.p_mw   -- active power calculated by reactive power control for DER-Inverter
            self.q_mvar -- reactive power calculated by reactive power control for DER-Inverter
        """
        # Calculate reactive power according to Q=f(V) Characteristic curve
        q_out = self._calculate_q_out(grid)
        # Normalize the momentary active power output to its max. value.
        mom_p_mw = self.p_mw
        self.norm_mom_p_mw = mom_p_mw / self.max_p_mw
        # Ensure normalized reactive power is within technical inverter range.
        self.norm_q_mvar = max(self.norm_min_q_mvar, min(q_out, self.norm_max_q_mvar))
        # Calculation normalized active power by given normalized apparent power and reactive power.
        self.norm_p_mw = min(self.norm_mom_p_mw, np.sqrt(1 - self.norm_q_mvar**2))
        # Calculate momentary apparent power by given normalized real power and reactive power.
        self.mom_sn_mva = np.sqrt(self.norm_p_mw**2 + self.norm_q_mvar**2)
        # Check Low Power Range Condition
        if np.abs(self.norm_mom_p_mw) <= self.reactive_power_performance_limit:
            # Store the unbounded reactive power value
            self.raw_norm_q_mvar = self.norm_q_mvar

            # Regulate reactive power to be no more than 10% of max active power, as in VDE.
            # low-power range according to VDE-AR-N 4105:2026-03: |Q| <= 10 % of P_Emax (= rated P), no Q when idle
            max_bounded_q_mvar = 0.10 * self.max_p_mw / self.sn_mva if mom_p_mw != 0 else 0.0
            q_mvar_abs = min(abs(self.norm_q_mvar), abs(max_bounded_q_mvar))

            # Update the reactive power and normalize respecting the limitation.
            self.q_mvar = np.copysign(q_mvar_abs, self.norm_q_mvar) * self.sn_mva
            self.norm_q_mvar = self.q_mvar / self.sn_mva

            # Update real power with the condition that it should not exceed the square root
            # of the difference of the square of apparent power and the square of reactive power.
            self.p_mw = min(mom_p_mw, np.sqrt(self.sn_mva**2 - self.q_mvar**2))
        else:
            # Active power exceeds 10% or 20% of the maximum active power.
            # Store the unbounded reactive power value
            self.raw_norm_q_mvar = self.norm_q_mvar
            # Q is determined by voltage (V_CCP) in voltage-reactive power mode (Q = f(V)).
            # Calculate max possible P using this Q and known S, considering available P.
            self.q_mvar = self.norm_q_mvar * self.sn_mva
            self.p_mw = min(mom_p_mw, np.sqrt(self.sn_mva**2 - self.q_mvar**2))

    def constant_power_factor_control(self, grid, cos_phi):
        """Fixed power factor according to VDE-AR-N 4105:2026-03.

        The active power is kept and the reactive power Q = |P| tan(phi) is taken from the headroom of the inverter
        rating S_n; |P| is reduced only if sqrt(P^2 + Q^2) would exceed S_n (as in voltage_reactive_q_control).
        Low-power range (|P| / S_n <= 0.2): |Q| <= 10 % of P_Emax. Idle storage provides no reactive power.

        Convention: storage table of pandapower = load convention, P > 0 charging; underexcited (absorption) +q,
        overexcited (injection) -q, independent of charging or discharging.
        """
        if cos_phi < self.min_cos_phi or cos_phi > 1:
            raise ValueError("cos_phi must be in the range [0.90, 1]")
        self.mom_cos_phi = cos_phi
        p_mw = self.p_mw
        mag = min(abs(p_mw), self.sn_mva)
        if mag <= 0.0:
            # idle storage: no reactive power
            self.q_mvar = 0.0
            self.norm_q_mvar = 0.0
            return
        tan_phi = np.sqrt(max(1.0 - cos_phi**2, 0.0)) / cos_phi
        q_abs = min(mag * tan_phi, self.norm_max_q_mvar * self.sn_mva)
        if mag / self.sn_mva <= self.reactive_power_performance_limit:
            # low-power range: at most 10 % of P_Emax
            q_abs = min(q_abs, 0.10 * self.max_p_mw)
        # sign of Q from the excitation mode only (load convention)
        self.q_mvar = self.get_q_sign() * q_abs
        self.norm_q_mvar = self.q_mvar / self.sn_mva
        self.p_mw = float(np.copysign(min(mag, np.sqrt(max(self.sn_mva**2 - self.q_mvar**2, 0.0))), p_mw))
        self.norm_p_mw = self.p_mw / self.sn_mva

    def _pq_from_cos_phi(self, s_mva, cos_phi):
        """Calculates active and reactive power from power factor."""
        # Get the signs for reactive and active power modes
        p_sign = self.get_p_sign()
        q_sign = self.get_q_sign()
        # Calculate active power
        p_mw = s_mva * cos_phi
        # Calculate reactive power
        q_mvar = p_sign * q_sign * np.sqrt(s_mva**2 - p_mw**2)
        return p_mw, q_mvar

    def _calculate_q_out(self, grid):
        """Reactive power setpoint of the Q(V) characteristic for the measured voltage at the own bus.

        Convention: storage table of pandapower = load convention, +q = absorption (underexcited, lowers the
        voltage), -q = injection (overexcited, raises the voltage).
        VDE-AR-N 4105: under-voltage -> overexcited (-q), over-voltage -> underexcited (+q); dead band between v_2
        and v_4, linear between v_1 and v_2 and between v_4 and v_5, saturated outside.

        Returns:
            q_out (float): reactive power normalised to S_n (load convention).
        """
        self.vm_pu_meas = grid.res_bus.at[self.bus, "vm_pu"]
        if self.vm_pu_meas <= self.v_1:
            # strong under-voltage: overexcited, saturated (-q_max)
            q_out = -self.norm_max_q_mvar
            self.q_mode = "overexcited"
        elif self.vm_pu_meas <= self.v_2:
            # under-voltage: overexcited, linear from -q_max at v_1 to 0 at v_2
            self.q_mode = "overexcited"
            m_12 = (self.nomi_q_mvar - self.norm_max_q_mvar) / (self.v_2 - self.v_1)
            v_dev = self.vm_pu_meas - self.v_1
            q_out = -(m_12 * v_dev + self.norm_max_q_mvar)
        elif self.vm_pu_meas <= self.v_4:
            # dead band: no reactive power
            q_out = self.nomi_q_mvar
        elif self.vm_pu_meas <= self.v_5:
            # over-voltage: underexcited, linear from 0 at v_4 to +q_max at v_5
            self.q_mode = "underexcited"
            m_45 = (self.norm_min_q_mvar - self.nomi_q_mvar) / (self.v_5 - self.v_4)
            v_dev = self.vm_pu_meas - self.v_4
            q_out = -(m_45 * v_dev + self.nomi_q_mvar)
        else:
            # strong over-voltage: underexcited, saturated (+q_max)
            self.q_mode = "underexcited"
            q_out = -self.norm_min_q_mvar
        return q_out

    def get_p_sign(self):
        if self.p_mw >= 0:
            return -1
        return 1

    def get_q_sign(self):
        if self.q_mode == "underexcited":
            return 1
        return -1

    def rule_based_p_control(self, grid):
        """
        Implements a rule-based control strategy for the battery storage system.
        The control strategy is based on the power generation and consumption
        at the same bus. It charges the battery if the state of charge (SoC)
        is below the charge limit and there is excess power generation. It
        discharges the battery if the SoC is above the discharge limit and
        there is a power deficit.
        """
        # Calculates the total power generation at the same bus
        pv_power = grid.sgen[grid.sgen["bus"] == self.bus]["p_mw"].sum()
        # Calculates the total power consumption at the same bus
        load_power = grid.load[grid.load["bus"] == self.bus]["p_mw"].sum()
        residual_power = pv_power - load_power
        # (I) Battery Charging Block:
        if (self.soc_percent < self.soc_percent_charge_limit) and (residual_power > 0):
            # positive charging
            # charge the surplus only
            p_mw = min(self.max_p_mw, residual_power)
        # (II) Battery Discharging Block:
        elif (self.soc_percent > self.soc_percent_discharge_limit) and (
            residual_power < 0
        ):
            # negative discharging
            # discharge the deficit only
            p_mw = max(self.min_p_mw, residual_power)
        else:
            p_mw = 0
        self.p_mw = p_mw
