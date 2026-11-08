"""Tests of the device controllers of egridlv_gym (PV, battery BES, EV charger, heat pump) beyond Q(U).

Expected behaviour (physics, VDE-AR-N 4105:2026-03 where it applies, and what the environment documents)
  PV   action -> cos phi in [0.90, 1] (action > 0 overexcited), fixed cos phi: P kept, Q = P tan(phi) within
       Q_max and S_n, P reduced only at S_n, low-power range, physical direction of Q
  BES  P action within the rating and the energy content of one 15-min step, SOC bookkeeping (efficiency,
       self-discharge, clipping), fixed cos phi with physical direction in BOTH charging and discharging,
       rule-based control (self-consumption at the bus)
  EV   P action limits, absent -> no power and SOC 0, arrival SOC brackets, SOC bookkeeping, rule-based charging
  HP   action scaling [-1, 1] -> [0.5, 1.5] x previous load, power limits, penalty of the cumulative shift
The tests check SOC efficiency per direction, self-discharge, the cos phi sign while charging and discharging, the
rule-based charging and the low-power range.
Run:  python -m pytest -q tests/test_devices.py   (from the repository root)
"""
import numpy as np
import pandapower as pp
import pytest

from egridlv_gym.wrapper.controllers.ev_charger_controller import EVChargerController
from egridlv_gym.wrapper.controllers.heat_pump_controller import HeatPumpController
from egridlv_gym.wrapper.controllers.pv_controller import PVController
from egridlv_gym.wrapper.controllers.storage_controller import StorageController

SN = 0.03       # MVA
E_MAX = 0.06    # MWh (BES)
DT = 0.25       # h, egridlv_gym step (15 min)


# ----------------------------------------------------------------------------- fixtures
def net_with(element, p_load=0.0, p_pv=0.0):
    net = pp.create_empty_network()
    b0, b1 = pp.create_bus(net, 0.4), pp.create_bus(net, 0.4)
    pp.create_ext_grid(net, b0)
    pp.create_line_from_parameters(net, b0, b1, 0.5, 0.2, 0.4, 0, 0.3)
    pp.create_load(net, b1, p_mw=p_load, q_mvar=0.0, sn_mva=SN)
    pp.create_sgen(net, b1, p_mw=p_pv, q_mvar=0.0, sn_mva=SN)
    pp.create_storage(net, b1, p_mw=0.0, q_mvar=0.0, max_e_mwh=E_MAX, sn_mva=SN, soc_percent=0.5,
                      max_p_mw=SN, min_p_mw=-SN)
    net.storage["efficiency_percent"] = 0.9
    return net


def pv(net):
    return PVController(id=0, pv=net.sgen.loc[0], control_q=True)


def bes(net, **kw):
    kw.setdefault("control_p", True)
    c = StorageController(storage=net.storage.loc[0], id=0, **kw)
    c.soc_percent = 0.5
    return c


def ev(net):
    probs = np.zeros(96)
    c = EVChargerController(storage=net.storage.loc[0], id=0, arrival_probabilities=probs,
                            departure_probabilities=probs.copy())
    c.reset(net)
    return c


def v_end_with(element, q_mvar, p_mw=0.0):
    """Feeder-end voltage with the given (p, q) written to the element table."""
    net = net_with(element)
    tab = net.sgen if element == "pv" else net.storage
    tab.at[0, "p_mw"], tab.at[0, "q_mvar"] = p_mw, q_mvar
    pp.runpp(net)
    return float(net.res_bus.vm_pu.iloc[1])


# ============================================================================= PV
@pytest.mark.parametrize("action,cos_phi,mode", [(0.0, 1.0, "underexcited"), (1.0, 0.9, "overexcited"),
                                                 (-1.0, 0.9, "underexcited"), (0.5, 0.95, "overexcited"),
                                                 (-0.5, 0.95, "underexcited")])
def test_pv_action_to_cos_phi(action, cos_phi, mode):
    net = net_with("pv")
    c = pv(net)
    c.p_mw = 0.5 * SN
    c.q_action(net, action)
    assert c.mom_cos_phi == pytest.approx(cos_phi)
    assert c.q_mode == mode


@pytest.mark.parametrize("cos_phi", [0.90, 0.95, 0.99, 1.0])
def test_pv_cos_phi_keeps_p_and_sets_q(cos_phi):
    """Below the rating the active power is kept and |Q| = P tan(phi)."""
    net = net_with("pv")
    c = pv(net)
    c.p_mw = 0.6 * SN
    c.q_mode = "underexcited"
    c.constant_power_factor_control(net, cos_phi)
    assert c.p_mw == pytest.approx(0.6 * SN)
    assert abs(c.q_mvar) == pytest.approx(0.6 * SN * np.tan(np.arccos(cos_phi)), rel=1e-6)


def test_pv_cos_phi_reduces_p_only_at_rating():
    net = net_with("pv")
    c = pv(net)
    c.p_mw = 1.0 * SN
    c.q_mode = "underexcited"
    c.constant_power_factor_control(net, 0.9)
    assert np.hypot(c.p_mw, c.q_mvar) <= SN * (1 + 1e-9)
    assert c.p_mw == pytest.approx(np.sqrt(SN ** 2 - c.q_mvar ** 2))
    assert abs(c.q_mvar) <= c.norm_max_q_mvar * SN + 1e-12


def test_pv_cos_phi_low_power_range():
    """Low-power range according to VDE-AR-N 4105:2026-03: |Q| <= 10 % of P_Emax (= S_n)."""
    net = net_with("pv")
    c = pv(net)
    c.p_mw = 0.1 * SN
    c.q_mode = "underexcited"
    c.constant_power_factor_control(net, 0.9)
    assert abs(c.q_mvar) <= 0.10 * c.max_p_mw + 1e-12
    assert abs(c.q_mvar) == pytest.approx(0.1 * SN * np.tan(np.arccos(0.9)))  # P tan(phi) is below that limit here


def test_pv_cos_phi_out_of_range_raises():
    net = net_with("pv")
    c = pv(net)
    c.p_mw = 0.5 * SN
    with pytest.raises(ValueError):
        c.constant_power_factor_control(net, 0.85)


@pytest.mark.parametrize("action,direction", [(1.0, +1), (-1.0, -1)])
def test_pv_cos_phi_physical_direction(action, direction):
    """Overexcited (action > 0) must raise, underexcited (action < 0) must lower the voltage (sgen, generator convention)."""
    net = net_with("pv")
    c = pv(net)
    c.p_mw = 0.6 * SN
    c.q_action(net, action)
    dv = v_end_with("pv", c.q_mvar, c.p_mw) - v_end_with("pv", 0.0, c.p_mw)
    assert np.sign(dv) == direction


# ============================================================================= BES
@pytest.mark.parametrize("action", np.linspace(-1, 1, 9))
def test_bes_p_action_within_rating(action):
    net = net_with("bes")
    c = bes(net)
    c.p_action(net, action)
    assert abs(c.p_mw) <= c.max_p_mw + 1e-12


def test_bes_p_action_energy_limit_and_penalty():
    net = net_with("bes")
    c = bes(net)
    c.soc_percent = 0.99                     # almost full: at most (E - e) / (eta dt) may be drawn
    pen = c.p_action(net, 1.0)
    assert pen == 1
    assert c.p_mw == pytest.approx((E_MAX - 0.99 * E_MAX) / (0.9 * DT), abs=1e-4)
    c.soc_percent = 0.01                     # almost empty: at most e * eta / dt may be delivered
    pen = c.p_action(net, -1.0)
    assert pen == 1
    assert c.p_mw == pytest.approx(-(0.01 * E_MAX) * 0.9 / DT, abs=1e-4)


def test_bes_p_action_headroom_is_exactly_usable():
    """Charging at the p_action limit fills the battery to 100 %, discharging at the limit empties it (no clipping)."""
    net = net_with("bes")
    c = bes(net)
    c.discharge_loss = 0.0
    c.soc_percent = 0.98
    c.p_action(net, 1.0)
    c.update_soc_from_previous(net)
    assert c.soc_percent == pytest.approx(1.0, abs=2e-3)
    c.soc_percent = 0.02
    c.p_action(net, -1.0)
    c.update_soc_from_previous(net)
    assert c.soc_percent == pytest.approx(0.0, abs=2e-3)


def test_bes_discharge_removes_p_over_eta():
    """Discharging P for 15 min removes P * dt / eta from the battery (plus self-discharge)."""
    net = net_with("bes")
    c = bes(net)
    c.discharge_loss = 0.0
    c.p_mw = -0.01
    c.update_soc_from_previous(net)
    assert c.soc_percent * E_MAX == pytest.approx(0.5 * E_MAX - 0.01 * DT / 0.9, rel=1e-9)


def test_bes_charge_stores_p_times_eta():
    net = net_with("bes")
    c = bes(net)
    c.discharge_loss = 0.0
    c.p_mw = 0.01
    c.update_soc_from_previous(net)
    assert c.soc_percent * E_MAX == pytest.approx(0.5 * E_MAX + 0.01 * DT * 0.9, rel=1e-9)


def test_bes_round_trip_efficiency_below_one():
    net = net_with("bes")
    c = bes(net)
    c.discharge_loss = 0.0
    e0 = c.soc_percent * E_MAX
    c.p_mw = 0.01                            # charge 0.01 MW for 15 min from the grid
    c.update_soc_from_previous(net)
    grid_in = 0.01 * DT
    stored = c.soc_percent * E_MAX - e0
    c.p_mw = -stored / DT * 0.9              # discharge what is stored, delivered energy = stored * eta
    c.update_soc_from_previous(net)
    grid_out = -c.p_mw * DT
    assert grid_out / grid_in == pytest.approx(0.81, abs=0.02)


def test_bes_self_discharge_value():
    """Implemented self-discharge: 0.1 % of the stored energy per day (3 %/month, assumption)."""
    net = net_with("bes")
    c = bes(net)
    c.p_mw = 0.0
    for _ in range(96):
        c.update_soc_from_previous(net)
    assert c.soc_percent == pytest.approx(0.5 * (1 - 0.001 / 96) ** 96, rel=1e-9)


def test_bes_self_discharge_plausible():
    net = net_with("bes")
    c = bes(net)
    c.p_mw = 0.0
    for _ in range(96):
        c.update_soc_from_previous(net)
    assert 1 - c.soc_percent / 0.5 < 0.01    # < 1 % per day


def test_bes_soc_clipped():
    net = net_with("bes")
    c = bes(net)
    c.soc_percent, c.p_mw = 0.99, SN
    c.update_soc_from_previous(net)
    assert c.soc_percent <= 1.0
    c.soc_percent, c.p_mw = 0.01, -SN
    c.update_soc_from_previous(net)
    assert c.soc_percent >= 0.0


def test_bes_idle_no_reactive_power():
    net = net_with("bes")
    c = bes(net)
    c.p_mw = 0.0
    c.q_action(net, 1.0)
    assert c.q_mvar == 0.0


@pytest.mark.parametrize("action,direction", [(1.0, +1), (-1.0, -1)])
def test_bes_cos_phi_physical_direction_discharging(action, direction):
    """Discharging (P < 0): overexcited must raise, underexcited must lower the voltage."""
    net = net_with("bes")
    c = bes(net)
    c.p_mw = -0.6 * SN
    c.q_action(net, action)
    dv = v_end_with("bes", c.q_mvar, c.p_mw) - v_end_with("bes", 0.0, c.p_mw)
    assert np.sign(dv) == direction


@pytest.mark.parametrize("action,direction", [(1.0, +1), (-1.0, -1)])
def test_bes_cos_phi_physical_direction_charging(action, direction):
    net = net_with("bes")
    c = bes(net)
    c.p_mw = 0.6 * SN
    c.q_action(net, action)
    dv = v_end_with("bes", c.q_mvar, c.p_mw) - v_end_with("bes", 0.0, c.p_mw)
    assert np.sign(dv) == direction


def test_bes_rule_based_charges_with_surplus_and_discharges_on_deficit():
    net = net_with("bes", p_load=0.005, p_pv=0.02)
    c = bes(net, control_p=False)
    c.rule_based_p_control(net)
    assert c.p_mw > 0
    net = net_with("bes", p_load=0.02, p_pv=0.0)
    c = bes(net, control_p=False)
    c.rule_based_p_control(net)
    assert c.p_mw < 0


def test_bes_rule_based_charges_surplus_only():
    net = net_with("bes", p_load=0.005, p_pv=0.02)
    c = bes(net, control_p=False)
    c.rule_based_p_control(net)
    assert c.p_mw == pytest.approx(0.02 - 0.005)


def test_bes_rule_based_discharges_deficit_only():
    net = net_with("bes", p_load=0.02, p_pv=0.005)
    c = bes(net, control_p=False)
    c.rule_based_p_control(net)
    assert c.p_mw == pytest.approx(-(0.02 - 0.005))


# ============================================================================= EV charger
@pytest.mark.parametrize("action", np.linspace(-1, 1, 5))
def test_ev_p_action_limits(action):
    net = net_with("bes")
    c = ev(net)
    c.p_action(net, action)
    assert c.min_p_mw - 1e-12 <= c.p_mw <= c.max_p_mw + 1e-12 or abs(c.p_mw) <= c.max_p_mw + 1e-12
    assert net.storage.at[0, "p_mw"] == pytest.approx(c.p_mw)


def test_ev_absent_no_power_soc_zero():
    net = net_with("bes")
    c = ev(net)
    c.at_home = False
    c.run(net, timestep=0, p_action=1.0, random_numbers=(1.0, 0.0, 0.0))
    assert net.storage.at[0, "p_mw"] == 0
    assert c.soc_percent == 0


@pytest.mark.parametrize("bracket,lo,hi", [(0.2, 0.3, 0.5), (0.7, 0.5, 0.8), (0.95, 0.8, 1.0)])
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_ev_arrival_soc_brackets(bracket, lo, hi, value):
    net = net_with("bes")
    c = ev(net)
    c.at_home = False
    c.arrival_probabilities[:] = 1.0
    c.update_state(0, net, (0.0, bracket, value))
    assert c.at_home
    assert lo - 1e-12 <= c.soc_percent <= hi + 1e-12


def test_ev_departure_sets_soc_zero_and_out_of_service():
    net = net_with("bes")
    c = ev(net)
    c.departure_probabilities[:] = 1.0
    c.update_state(0, net, (0.0, 0.0, 0.0))
    assert not c.at_home and c.soc_percent == 0
    assert not net.storage.at[0, "in_service"]


def test_ev_soc_bookkeeping_lossless_documented():
    """Documents: the EV charger stores P * dt without losses."""
    net = net_with("bes")
    c = ev(net)
    c.p_mw = 0.004
    c.update_soc_from_previous(net)
    assert c.soc_percent == pytest.approx(0.5 + 0.004 * DT / c.max_e_mwh)


def test_ev_rule_based_charges_until_full():
    net = net_with("bes")
    c = ev(net)
    c.rule_based_p_control(net, 0)
    assert c.p_mw == pytest.approx(min(c.max_p_mw, (1 - 0.5) * c.max_e_mwh / DT))
    c.soc_percent = 1.0
    c.rule_based_p_control(net, 0)
    assert c.p_mw == 0


# ============================================================================= heat pump
@pytest.mark.parametrize("a,scale", [(-1.0, 0.5), (0.0, 1.0), (1.0, 1.5), (0.5, 1.25)])
def test_hp_action_scaling(a, scale):
    net = net_with("bes", p_load=0.004)
    c = HeatPumpController(load=net.load.loc[0], id=0)
    assert c.scale_action(a) == pytest.approx(scale)


@pytest.mark.parametrize("a", [-1.0, 0.0, 1.0])
def test_hp_power_follows_scaled_previous_load_within_limits(a):
    net = net_with("bes", p_load=0.004)
    c = HeatPumpController(load=net.load.loc[0], id=0)
    c.run(net, p_action=a, timestep=0)
    assert c.p_mw == pytest.approx(np.clip(c.scale_action(a) * 0.004, 0, c.max_p_mw))
    assert net.load.at[0, "p_mw"] == pytest.approx(c.p_mw)


def test_hp_power_clipped_at_rating():
    net = net_with("bes", p_load=0.9 * SN)
    c = HeatPumpController(load=net.load.loc[0], id=0)
    c.run(net, p_action=1.0, timestep=0)
    assert c.p_mw <= c.max_p_mw + 1e-12


def test_hp_penalty_for_cumulative_shift():
    """Shifting the load up repeatedly accumulates the error; above max_error (= rating) a penalty arises."""
    net = net_with("bes", p_load=0.6 * SN)
    c = HeatPumpController(load=net.load.loc[0], id=0)
    pens = []
    for _ in range(10):
        net.load.at[0, "p_mw"] = 0.6 * SN       # profile value of the next step
        pens.append(c.run(net, p_action=1.0, timestep=0))
    assert pens[0] == 0
    assert max(pens) > 0
