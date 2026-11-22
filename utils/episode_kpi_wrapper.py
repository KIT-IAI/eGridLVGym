"""
Physical quantities of an episode for the training logs.

The wrapper sums the quantities below over an episode (one week of 672 steps) and adds them to the info of the last
step. VecMonitor(info_keywords=KPI_KEYS) then writes one column per quantity to the monitor files of the training and
of the periodic evaluation, next to the episode reward, so that learning curves can be drawn in physical units.

    grid_cost_eur          energy exchanged at the transformer x day-ahead price x 0.25 h (negative = net revenue)
    trafo_over100_steps    steps with transformer loading above 100 %
    line_over100_steps     steps with the highest line loading above 100 %
    v_out_095_105_steps    steps with a bus voltage outside 0.95-1.05 p.u.
    v_out_090_110_steps    steps with a bus voltage outside 0.90-1.10 p.u.
    trafo_max_pct, line_max_pct, v_min_pu, v_max_pu     extreme values of the episode
    ev_departures          number of vehicle departures
    ev_departure_soc       mean state of charge (0-1) at the last step before a departure (nan without departure)
    pv_curtailment_sum     sum of the curtailment values of the PV controllers over all steps
    device_penalties       sum of the device bound penalties (battery, charger, PV, heat pump)
    r_price, r_ev_flexibility, r_loading, r_voltage, r_curtailment   sums of the reward terms (0 when not in the mode)
"""
import gymnasium as gym
import numpy as np

REWARD_TERMS = ("price", "ev_flexibility", "loading", "voltage", "curtailment")
KPI_KEYS = ("grid_cost_eur", "trafo_over100_steps", "line_over100_steps", "v_out_095_105_steps", "v_out_090_110_steps",
            "trafo_max_pct", "line_max_pct", "v_min_pu", "v_max_pu", "ev_departures", "ev_departure_soc",
            "pv_curtailment_sum", "device_penalties") + tuple("r_" + k for k in REWARD_TERMS)
PENALTIES = ("battery_penalty", "charger_penalty", "pv_penalty", "hp_penalty")


class EpisodeKpiWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # as base_gym._current_price(); info["timestep"] is the index after the step, the price of the step is one before
        self._price = self.env.unwrapped.prices["price_clipped"].to_numpy() * 10.0
        self._clear()

    def _clear(self):
        self._k = dict.fromkeys(KPI_KEYS, 0.0)
        self._k.update(v_min_pu=np.inf, v_max_pu=-np.inf)
        self._soc_sum, self._present, self._soc = 0.0, None, None

    def reset(self, **kwargs):
        self._clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        k = self._k
        v = np.asarray(info["voltage"], dtype=float)
        trafo = float(np.max(info["trafo_loading"])) if np.size(info["trafo_loading"]) else 0.0
        line = float(np.max(info["line_loading"])) if np.size(info["line_loading"]) else 0.0
        k["grid_cost_eur"] += float(np.sum(info["ext_power"])) * self._price[int(info["timestep"]) - 1] * 0.25
        k["trafo_over100_steps"] += trafo > 100.0
        k["line_over100_steps"] += line > 100.0
        k["v_out_095_105_steps"] += bool(v.min() < 0.95 or v.max() > 1.05)
        k["v_out_090_110_steps"] += bool(v.min() < 0.90 or v.max() > 1.10)
        k["trafo_max_pct"], k["line_max_pct"] = max(k["trafo_max_pct"], trafo), max(k["line_max_pct"], line)
        k["v_min_pu"], k["v_max_pu"] = min(k["v_min_pu"], float(v.min())), max(k["v_max_pu"], float(v.max()))
        k["pv_curtailment_sum"] += float(np.sum(info["pv_curtailment"]))
        k["device_penalties"] += float(sum(np.sum(info[p]) for p in PENALTIES))
        for term in REWARD_TERMS:
            k["r_" + term] += float(info.get("reward_" + term, 0.0))
        present = np.asarray(info["ev_present"], dtype=bool)
        soc = np.asarray(info["ev_soc_percent"], dtype=float)
        if self._present is not None and present.size:
            left = self._present & ~present
            k["ev_departures"] += int(left.sum())
            self._soc_sum += float(self._soc[left].sum())
        self._present, self._soc = present, soc
        if terminated or truncated:
            out = {key: float(val) for key, val in k.items()}
            out["ev_departure_soc"] = self._soc_sum / k["ev_departures"] if k["ev_departures"] else float("nan")
            info = dict(info, **out)
        return obs, reward, terminated, truncated, info
