import numpy as np



class PVController:
    def __init__(
        self,
        id,
        pv,
        power_scaling=1,
        control_q=False,
        control_q_shielding=False,
    ):
        self.power_scaling = power_scaling
        self.id = id
        self.pv = pv
        self.bus = pv.bus
        self.sn_mva = pv.sn_mva * power_scaling
        self.max_p_mw = self.sn_mva
        self.c_precision = 4
        self.p_sign = -1
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

        # Reference voltage values for the Q(V) characteristic (deadband)
        self.v_1 = 0.93
        self.v_2 = 0.97
        self.v_4 = 1.03
        self.v_5 = 1.07

        # Nominal reactive power value
        self.nomi_q_mvar = 0

        self.control_q = control_q
        self.control_q_shielding = control_q_shielding

        self.p_mw = 0
        self.q_mvar = 0
        self.unadjusted_power = 0
        self.q_mode = "underexcited"

    def reset(self, grid_state):
        self.p_mw = grid_state.sgen.at[self.id, "p_mw"]
        self.q_mvar = 0
        # reset the reactive power of this PV unit (sgen)
        grid_state.sgen.at[self.id, "q_mvar"] = self.q_mvar
        self.q_mode = "underexcited"

    def run(self, grid_state, timestep, p_action=None, q_action=None):
        self.p_mw = grid_state.sgen.at[self.id, "p_mw"]
        self.unadjusted_power = self.p_mw
        if self.control_q:
            self.q_action(grid_state, q_action)
            # write the reactive power of the power factor law to the grid (generator convention, negative = underexcited)
            grid_state.sgen.at[self.id, "q_mvar"] = self.q_mvar

        grid_state.sgen.at[self.id, "p_mw"] = self.p_mw

        return 0

    def _p_action(self, grid_state, timestep, p_action):
        pass

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

    def constant_power_factor_control(self, grid, cos_phi):
        """Fixed power factor according to VDE-AR-N 4105:2026-03.

        The active power is kept and the reactive power Q = P tan(phi) is taken from the headroom of the inverter
        rating S_n; P is reduced only if sqrt(P^2 + Q^2) would exceed S_n (as in voltage_reactive_q_control).
        Low-power range (P / S_n <= 0.2): |Q| <= 10 % of P_Emax.

        Convention: sgen table of pandapower = generator convention; overexcited (injection) +q, underexcited
        (absorption) -q.
        """
        if cos_phi < self.min_cos_phi or cos_phi > 1:
            raise ValueError("cos_phi must be in the range [0.90, 1]")
        self.mom_cos_phi = cos_phi
        # feed-in limited to the inverter rating
        p_av = min(self.p_mw, self.sn_mva)
        tan_phi = np.sqrt(max(1.0 - cos_phi**2, 0.0)) / cos_phi
        q_abs = min(p_av * tan_phi, self.norm_max_q_mvar * self.sn_mva)
        if p_av / self.sn_mva <= self.reactive_power_performance_limit:
            # low-power range: at most 10 % of P_Emax
            q_abs = min(q_abs, 0.10 * self.max_p_mw)
        # sign of Q from the excitation mode (generator convention)
        self.q_mvar = self.p_sign * self.get_q_sign() * q_abs
        self.norm_q_mvar = self.q_mvar / self.sn_mva
        self.p_mw = min(p_av, np.sqrt(max(self.sn_mva**2 - self.q_mvar**2, 0.0)))
        self.norm_p_mw = self.p_mw / self.sn_mva

    def voltage_reactive_q_control(self, grid_state):
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
            net --  egrid
            mom_p_mw -- momentary active power desired by run_battery_p_control
        Returns:
            self.p_mw   -- active power calculated by reactive power control for DER-Inverter
            self.q_mvar -- reactive power calculated by reactive power control for DER-Inverter
        """
        # Calculate reactive power according to Q=f(V) Characteristic curve
        q_out = self._calculate_q_out(grid_state)
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
            # low-power range according to VDE-AR-N 4105:2026-03: |Q| <= 10 % of P_Emax (= S_n here), reactive power
            # only during feed-in (no Q at P = 0)
            max_bounded_q_mvar = 0.10 * self.max_p_mw / self.sn_mva if mom_p_mw > 0 else 0.0
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

        grid_state.sgen.at[self.id, "p_mw"] = self.p_mw
        grid_state.sgen.at[self.id, "q_mvar"] = self.q_mvar

    def get_q_sign(self):
        """Calculates the Q-Sign value based on the given Q mode.
        Args:
            qmode (str): The Q mode "ind"("underexcited") or "cap"("overexcited").
        Returns:
            qsign (bool): The Q-Sign value."""
        if self.q_mode in ("ind", "underexcited"):
            qsign = 1
        elif self.q_mode in ("cap", "overexcited"):
            qsign = -1
        else:
            raise ValueError(
                f'Unknown mode {self.q_mode} - specify "underexcited" (Q absorption, decreases voltage)'
                f' or "overexcited" (Q injection, increases voltage)'
            )
        return qsign

    def _calculate_q_out(self, grid):
        """Reactive power setpoint of the Q(V) characteristic for the measured voltage at the own bus.

        Convention: sgen table of pandapower = generator convention, +q = overexcited (injection, raises the
        voltage), -q = underexcited (absorption, lowers the voltage).
        VDE-AR-N 4105: under-voltage -> overexcited (+q), over-voltage -> underexcited (-q); dead band between v_2
        and v_4, linear between v_1 and v_2 and between v_4 and v_5, saturated outside.

        Returns:
            q_out (float): reactive power normalised to S_n.
        """
        self.vm_pu_meas = grid.res_bus.at[self.bus, "vm_pu"]
        if self.vm_pu_meas <= self.v_1:
            # strong under-voltage: overexcited, saturated (+q_max)
            self.q_mode = "overexcited"
            q_out = self.norm_max_q_mvar
        elif self.vm_pu_meas <= self.v_2:
            # under-voltage: overexcited, linear from +q_max at v_1 to 0 at v_2
            self.q_mode = "overexcited"
            m_12 = (self.nomi_q_mvar - self.norm_max_q_mvar) / (self.v_2 - self.v_1)
            v_dev = self.vm_pu_meas - self.v_1
            q_out = m_12 * v_dev + self.norm_max_q_mvar
        elif self.vm_pu_meas <= self.v_4:
            # dead band: no reactive power
            q_out = self.nomi_q_mvar
        elif self.vm_pu_meas <= self.v_5:
            # over-voltage: underexcited, linear from 0 at v_4 to -q_max at v_5
            self.q_mode = "underexcited"
            m_45 = (self.norm_min_q_mvar - self.nomi_q_mvar) / (self.v_5 - self.v_4)
            v_dev = self.vm_pu_meas - self.v_4
            q_out = m_45 * v_dev + self.nomi_q_mvar
        else:
            # strong over-voltage: underexcited, saturated (-q_max)
            self.q_mode = "underexcited"
            q_out = self.norm_min_q_mvar
        return q_out

    def get_curtailment(self):
        return self.unadjusted_power - self.p_mw

    def _pq_from_cos_phi(self, s_mva, cos_phi):
        """Calculates active and reactive power from power factor."""
        # Get the signs for reactive and active power modes
        p_sign = -1
        q_sign = self.get_q_sign()
        # Calculate active power
        p_mw = s_mva * cos_phi
        # Calculate reactive power
        q_mvar = p_sign * q_sign * np.sqrt(s_mva**2 - p_mw**2)
        return p_mw, q_mvar
