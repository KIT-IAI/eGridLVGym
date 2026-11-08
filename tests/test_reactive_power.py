"""Tests of the reactive power functions of the PV and battery (BES) controllers against VDE-AR-N 4105:2026-03.

What is checked
  1. parameters      Q(U) support points 0.93 / 0.97 / 1.03 / 1.07 p.u., Q_max from cos phi 0.90 (S_n > 4.6 kVA)
                     resp. 0.95 (small units)
  2. curve values    _calculate_q_out at and between the support points, dead band 0.97..1.03, saturation
  3. shape           continuity at the support points, monotonic in the voltage, symmetric around 1.0 p.u.
  4. convention      pandapower: sgen = generator convention (+q raises U), storage = load convention (+q lowers U)
  5. physics         the Q(U) output written to the element table moves the voltage TOWARDS the band
                     (under-voltage -> overexcited, over-voltage -> underexcited), for PV and BES
  6. full control    voltage_reactive_q_control: apparent power limit S_n (P reduced in favour of Q),
                     low power range, values written to the grid tables, q_mode labels
  7. closed loop     repeated Q(U) steps on a small feeder: PV reduces an over-voltage, BES lifts an under-voltage

Run:  python -m pytest -q tests/test_reactive_power.py   (from the repository root)
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pytest

from egridlv_gym.wrapper.controllers.pv_controller import PVController
from egridlv_gym.wrapper.controllers.storage_controller import StorageController

QMAX_090 = np.sin(np.arccos(0.90))  # 0.4359: Q_max / S_n for cos phi 0.90
QMAX_095 = np.sin(np.arccos(0.95))  # 0.3122: Q_max / S_n for cos phi 0.95
V_POINTS = (0.93, 0.97, 1.03, 1.07)
VOLTAGES = np.round(np.arange(0.88, 1.1201, 0.005), 4)


# ----------------------------------------------------------------------------- helpers
def feeder(element: str, sn_mva: float = 0.03, length_km: float = 0.5):
    """Two-bus LV feeder (0.4 kV, ext grid at bus 0) with one PV (sgen) or BES (storage) at the end (bus 1)."""
    net = pp.create_empty_network()
    b0, b1 = pp.create_bus(net, 0.4), pp.create_bus(net, 0.4)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_line_from_parameters(net, b0, b1, length_km, r_ohm_per_km=0.2, x_ohm_per_km=0.4, c_nf_per_km=0,
                                   max_i_ka=0.3)
    if element == "pv":
        pp.create_sgen(net, b1, p_mw=0.0, q_mvar=0.0, sn_mva=sn_mva)
    else:
        pp.create_storage(net, b1, p_mw=0.0, q_mvar=0.0, max_e_mwh=2 * sn_mva, sn_mva=sn_mva, soc_percent=0.5,
                          max_p_mw=sn_mva, min_p_mw=-sn_mva)
        net.storage["efficiency_percent"] = 0.9  # read by StorageController, not created by pandapower
    pp.create_load(net, b1, p_mw=0.0, q_mvar=0.0)
    return net


def make_pv(net, sn_mva=0.03):
    return PVController(id=0, pv=net.sgen.loc[0], control_q=True)


def make_bes(net):
    return StorageController(storage=net.storage.loc[0], id=0, control_p=True)


def set_voltage(net, v):
    """Pretend a measured voltage at bus 1 (what _calculate_q_out reads)."""
    net.res_bus = pd.DataFrame({"vm_pu": [1.0, v]})


def expected_q_gen(v, qmax):
    """VDE-AR-N 4105 Q(U) curve, normalised to S_n, GENERATOR convention (+ = overexcited, raises the voltage)."""
    v1, v2, v4, v5 = V_POINTS
    if v <= v1:
        return qmax
    if v <= v2:
        return qmax * (v2 - v) / (v2 - v1)
    if v <= v4:
        return 0.0
    if v <= v5:
        return -qmax * (v - v4) / (v5 - v4)
    return -qmax


def voltage_effect(element, q_mvar):
    """Voltage change at the feeder end when q_mvar is written to the element table (no active power)."""
    net = feeder(element)
    tab = net.sgen if element == "pv" else net.storage
    tab.at[0, "q_mvar"] = q_mvar
    pp.runpp(net)
    return float(net.res_bus.vm_pu.iloc[1]) - 1.0


# ----------------------------------------------------------------------------- 1. parameters
@pytest.mark.parametrize("cls,element", [(PVController, "pv"), (StorageController, "bes")])
def test_support_points_and_qmax(cls, element):
    net = feeder(element)
    c = make_pv(net) if element == "pv" else make_bes(net)
    assert (c.v_1, c.v_2, c.v_4, c.v_5) == V_POINTS
    assert c.min_cos_phi == 0.9
    assert c.norm_max_q_mvar == pytest.approx(QMAX_090, abs=1e-4)  # rounded to 4 decimals (c_precision)
    assert c.norm_min_q_mvar == pytest.approx(-QMAX_090, abs=1e-4)


def test_small_pv_uses_cos_phi_095():
    net = feeder("pv", sn_mva=0.004)
    c = make_pv(net)
    assert c.min_cos_phi == 0.95
    assert c.norm_max_q_mvar == pytest.approx(QMAX_095, abs=1e-4)


# ----------------------------------------------------------------------------- 2. curve values
@pytest.mark.parametrize("v", VOLTAGES)
def test_pv_curve_values(v):
    """PV writes to sgen (generator convention): q_out must equal the VDE curve directly."""
    net = feeder("pv")
    c = make_pv(net)
    set_voltage(net, v)
    assert c._calculate_q_out(net) == pytest.approx(expected_q_gen(v, c.norm_max_q_mvar), abs=1e-9)


@pytest.mark.parametrize("v", VOLTAGES)
def test_bes_curve_values(v):
    """BES writes to storage (load convention): q_out must equal MINUS the VDE curve in generator convention."""
    net = feeder("bes")
    c = make_bes(net)
    set_voltage(net, v)
    assert c._calculate_q_out(net) == pytest.approx(-expected_q_gen(v, c.norm_max_q_mvar), abs=1e-9)


@pytest.mark.parametrize("element", ["pv", "bes"])
@pytest.mark.parametrize("v", [0.97, 0.99, 1.00, 1.01, 1.03])
def test_dead_band(element, v):
    net = feeder(element)
    c = make_pv(net) if element == "pv" else make_bes(net)
    set_voltage(net, v)
    assert c._calculate_q_out(net) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("element", ["pv", "bes"])
@pytest.mark.parametrize("v", [0.85, 0.90, 0.93, 1.07, 1.10, 1.20])
def test_saturation(element, v):
    net = feeder(element)
    c = make_pv(net) if element == "pv" else make_bes(net)
    set_voltage(net, v)
    assert abs(c._calculate_q_out(net)) == pytest.approx(c.norm_max_q_mvar, abs=1e-9)


# ----------------------------------------------------------------------------- 3. shape
@pytest.mark.parametrize("element", ["pv", "bes"])
def test_continuous_and_monotonic(element):
    net = feeder(element)
    c = make_pv(net) if element == "pv" else make_bes(net)
    vs = np.linspace(0.85, 1.15, 3001)
    q = []
    for v in vs:
        set_voltage(net, v)
        q.append(c._calculate_q_out(net))
    q = np.array(q) * (1 if element == "pv" else -1)  # both in generator convention
    assert np.all(np.diff(q) <= 1e-12), "Q (generator convention) must not increase with the voltage"
    assert np.max(np.abs(np.diff(q))) < 0.005, "no jumps at the support points"


@pytest.mark.parametrize("element", ["pv", "bes"])
@pytest.mark.parametrize("dv", [0.01, 0.03, 0.05, 0.08])
def test_symmetric_around_nominal(element, dv):
    net = feeder(element)
    c = make_pv(net) if element == "pv" else make_bes(net)
    set_voltage(net, 1.0 - dv)
    q_low = c._calculate_q_out(net)
    set_voltage(net, 1.0 + dv)
    q_high = c._calculate_q_out(net)
    assert q_low == pytest.approx(-q_high, abs=1e-9)


# ----------------------------------------------------------------------------- 4. pandapower conventions
def test_pandapower_sign_conventions():
    """The basis of the curve signs: +q on an sgen raises, +q on a storage lowers the voltage."""
    assert voltage_effect("pv", +0.01) > 0
    assert voltage_effect("pv", -0.01) < 0
    assert voltage_effect("bes", +0.01) < 0
    assert voltage_effect("bes", -0.01) > 0


# ----------------------------------------------------------------------------- 5. physics: towards the band
@pytest.mark.parametrize("element", ["pv", "bes"])
@pytest.mark.parametrize("v", [0.90, 0.93, 0.95, 1.05, 1.07, 1.10])
def test_q_moves_voltage_towards_band(element, v):
    """Outside the dead band the curve output must move the voltage towards 1.0 p.u.:
    under-voltage -> overexcited (raise U), over-voltage -> underexcited (lower U)."""
    net = feeder(element)
    c = make_pv(net) if element == "pv" else make_bes(net)
    set_voltage(net, v)
    q = c._calculate_q_out(net) * 0.01  # scaled Q in Mvar
    dv = voltage_effect(element, q)
    assert np.sign(dv) == np.sign(1.0 - v), f"{element}: at V = {v} the Q(U) output moves the voltage away from the band"


@pytest.mark.parametrize("element", ["pv", "bes"])
@pytest.mark.parametrize("v,mode", [(0.92, "overexcited"), (0.95, "overexcited"), (1.05, "underexcited"),
                                    (1.09, "underexcited")])
def test_q_mode_label(element, v, mode):
    net = feeder(element)
    c = make_pv(net) if element == "pv" else make_bes(net)
    set_voltage(net, v)
    c._calculate_q_out(net)
    assert c.q_mode == mode


# ----------------------------------------------------------------------------- 6. full Q(U) control
def run_pv_qu(v, p_rel, sn=0.03):
    net = feeder("pv", sn_mva=sn)
    c = make_pv(net)
    set_voltage(net, v)
    c.p_mw = p_rel * c.max_p_mw
    c.voltage_reactive_q_control(net)
    return net, c


@pytest.mark.parametrize("v", [0.92, 1.08])
def test_pv_full_power_apparent_limit(v):
    """At P = S_n and saturated Q, P is reduced to sqrt(S_n^2 - Q^2) (VDE: P may be reduced in favour of Q)."""
    net, c = run_pv_qu(v, 1.0)
    assert abs(c.q_mvar) == pytest.approx(c.norm_max_q_mvar * c.sn_mva, rel=1e-6)
    assert c.p_mw == pytest.approx(np.sqrt(c.sn_mva ** 2 - c.q_mvar ** 2), rel=1e-6)
    assert np.hypot(c.p_mw, c.q_mvar) <= c.sn_mva * (1 + 1e-9)
    assert net.sgen.at[0, "q_mvar"] == pytest.approx(c.q_mvar)
    assert net.sgen.at[0, "p_mw"] == pytest.approx(c.p_mw)


def test_pv_full_power_written_q_has_vde_sign():
    net, c = run_pv_qu(0.92, 1.0)
    assert net.sgen.at[0, "q_mvar"] > 0      # under-voltage -> overexcited -> injection
    net, c = run_pv_qu(1.08, 1.0)
    assert net.sgen.at[0, "q_mvar"] < 0      # over-voltage -> underexcited -> absorption


@pytest.mark.parametrize("p_rel", [0.05, 0.10, 0.15, 0.20])
@pytest.mark.parametrize("v", [0.92, 1.08])
def test_pv_low_power_range(p_rel, v):
    """Low-power range according to VDE-AR-N 4105:2026-03: for P/P_Emax <= 0.2, |Q| <= 10 % of P_Emax (= S_n)."""
    net, c = run_pv_qu(v, p_rel)
    assert abs(c.q_mvar) == pytest.approx(0.10 * c.max_p_mw, rel=1e-6)


@pytest.mark.parametrize("v", [0.92, 1.08])
def test_pv_no_reactive_power_without_feed_in(v):
    """According to VDE-AR-N 4105:2026-03 reactive power is provided during feed-in only; P = 0 -> Q = 0."""
    net, c = run_pv_qu(v, 0.0)
    assert c.q_mvar == 0.0


def test_pv_low_power_range_norm_pemax():
    net, c = run_pv_qu(1.08, 0.05)
    assert abs(c.q_mvar) == pytest.approx(0.10 * c.max_p_mw, rel=1e-6)


def test_bes_full_control_writes_storage_q_with_vde_sign():
    for v, sign in ((0.92, -1), (1.08, +1)):  # storage table: load convention
        net = feeder("bes")
        c = make_bes(net)
        set_voltage(net, v)
        c.p_mw = 0.8 * c.max_p_mw  # charging, outside the low power range
        c.vde_adjust(net, "voltage_reactive")
        assert np.sign(net.storage.at[0, "q_mvar"]) == sign
        assert np.hypot(c.p_mw, c.q_mvar) <= c.sn_mva * (1 + 1e-9)


# ----------------------------------------------------------------------------- 7. closed loop on a feeder
def shielding_pass(element, p_mw, load_mw, passes=1):
    """As GridManager.step with control_q_shielding: power flow -> Q(U) from that voltage -> power flow, 'passes' times.
    Returns the feeder-end voltages [before, after pass 1, after pass 2, ...]."""
    net = feeder(element, sn_mva=0.05, length_km=1.0)
    net.load.at[0, "p_mw"] = load_mw
    tab = net.sgen if element == "pv" else net.storage
    tab.at[0, "p_mw"] = p_mw
    c = make_pv(net) if element == "pv" else make_bes(net)
    pp.runpp(net)
    vs = [float(net.res_bus.vm_pu.iloc[1])]
    for _ in range(passes):
        c.p_mw = p_mw
        if element == "pv":
            c.voltage_reactive_q_control(net)
        else:
            c.vde_adjust(net, "voltage_reactive")
        pp.runpp(net)
        vs.append(float(net.res_bus.vm_pu.iloc[1]))
    return vs


def test_shielding_pass_pv_reduces_overvoltage():
    v = shielding_pass("pv", p_mw=0.045, load_mw=0.0)
    assert v[0] > 1.03, "test case must start in the over-voltage part of the curve"
    assert v[1] < v[0]


def test_shielding_pass_bes_lifts_undervoltage():
    v = shielding_pass("bes", p_mw=0.040, load_mw=0.02)  # charging + load -> under-voltage
    assert v[0] < 0.97, "test case must start in the under-voltage part of the curve"
    assert v[1] > v[0]


def test_shielding_pass_bes_reduces_overvoltage_while_discharging():
    v = shielding_pass("bes", p_mw=-0.045, load_mw=0.0)  # discharging into an unloaded feeder
    assert v[0] > 1.03
    assert v[1] < v[0]


def test_undamped_repetition_oscillates_documented():
    """Documents a property, not a requirement: repeating the one-shot Q(U) of egridlv_gym without a PT1 / damping
    on a stiff feeder alternates between two points (Q pushes U into the dead band, Q -> 0, U jumps back).
    egridlv_gym applies one pass per 15-min step, so no oscillation arises inside the environment, but the result of
    that pass is not an equilibrium of the curve (grid codes require a PT1 behaviour of the Q(U) control for this reason, e.g. VDE-AR-N 4110:2023-09)."""
    v = shielding_pass("pv", p_mw=0.045, load_mw=0.0, passes=6)
    assert v[2] == pytest.approx(v[0], abs=1e-9) and v[4] == pytest.approx(v[0], abs=1e-9)
    assert v[1] == pytest.approx(v[3], abs=1e-9) and v[1] < v[0]
