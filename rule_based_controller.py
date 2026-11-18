import numpy as np


class RuleBasedController:
    """
    Rule based controllers that can be used as a baseline for the eGridLV environment.
    """

    def __init__(self, mode="baseline", ev_target=0.5, price_memory=None):
        """
        Initializes the rule-based controller.
        Args:
            mode: str, the rule the controller should follow.
                  "baseline"            no device action
                  "price_window"        current price vs. moving average of the last 96 prices (24 h look-back)
                  "price_forecast"      current price vs. mean of the 8-step price forecast (2 h look-ahead, no memory);
                                        same price information as the RL agents
                  "price_forecast_avg"  mean of the 8-step forecast vs. moving average of the last 8 forecast means
                                        (the former commented-out variant "#2"; 2 h look-ahead plus 2 h look-back)
            ev_target: float, target SOC for EVs.
            price_memory: list or None, historical price memory.
        """
        self.window = 29
        self.price_memory = [29 for _ in range(96)]
        self.mode = mode
        self.ev_target = ev_target

    def predict(self, observation, deterministic=True):
        """
        Emulates the predict method of a RL agent.
        """
        if self.mode in ("price_window", "price_forecast", "price_forecast_avg"):
            if self.mode == "price_window":
                price_state = self.decide_price_state_window(observation)
            elif self.mode == "price_forecast":
                price_state = self.decide_price_state_forecast(observation)
            else:
                price_state = self.decide_based_on_price_forecast(observation)
            # price_state = self.decide_based_on_price_forecast(observation) #2
            bes_action = self.choose_bes_actions(observation, price_state)
            charging_point_action = self.charger_actions(observation, price_state)
            hp_action = self.hp_actions(observation, price_state)
        elif self.mode == "baseline":
            bes_action = [0 for i in range(self.storages_count)]
            charging_point_action = [0 for i in range(self.ev_count)]
            hp_action = [0 for i in range(self.hp_count)]
        else:
            raise ValueError("Invalid rule")
        action = np.concatenate([bes_action, charging_point_action, hp_action]).reshape(
            1, -1
        )
        return action, None

    def charger_actions(self, observation, price_state):
        """
        Decides the action of the charger based on the current state of the EV.
        Will always charge the EV if the EV is present and the SOC is below the target.
        If the SOC is above the target, the charger will only charge the EV if the price is low.
        If the price is high it will discharge the EV defensively.
        """
        actions = []
        soc_observation = observation["ev_soc_percent"][0]
        ev_present = observation["ev_present"][0].flatten()
        for i in range(self.ev_count):
            soc_percent = soc_observation[i]
            charge = self.max_e_mwh_ev[i] * soc_percent
            max_e = self.max_e_mwh_ev[i]
            max_p = self.max_p_mw_ev[i]
            space = max_e - charge
            if ev_present[i] == 1:
                if (price_state == 1) and (soc_percent > 0.75):
                    action = max(-charge, -max_p)
                elif price_state == -1:
                    action = min(space, max_p)
                elif soc_percent < self.ev_target:
                    action = max_p / 2
                else:
                    action = 0
            else:
                action = 0
            action = action / max_p
            actions.append(action)
        return actions

    def hp_actions(self, observation, price_state):
        """
        Decides the action (load scaling) of the heat pump based on the current price state.
        """
        if price_state == 1:
            return [-1 for i in range(self.hp_count)]
        elif price_state == -1:
            return [1 for i in range(self.hp_count)]
        else:
            return [0 for i in range(self.hp_count)]

    def decide_price_state_window(self, observation):
        """
        Determines the next price based on a sliding window approach
        Updates the window and compares price to its moving average.
        """
        next_price = observation["price"].flatten()[0]
        self.price_memory.append(next_price)
        self.price_memory = self.price_memory[1:]
        self.window = sum(self.price_memory) / 96
        if next_price > self.window * 1.1:
            # high price
            return 1
        elif next_price < self.window * 0.9:
            # low price
            return -1
        else:
            # average price
            return 0

    def decide_price_state_forecast(self, observation):
        """
        Price state from the 8-step price forecast only (2 h look-ahead, no memory): the current price is compared
        with the mean of the forecast window, with the same 10 % thresholds as the price-window rule.
        """
        forecast = observation["price"].flatten()
        now, ref = forecast[0], np.mean(forecast)
        if now > ref * 1.1:
            return 1
        elif now < ref * 0.9:
            return -1
        return 0

    def decide_based_on_price_forecast(self, observation):
        """
        Determines the price state (high, average, low) based on a forecast of the next 8 prices.
        Compares the average forecast to a short moving average of recent forecasted averages.
        """
        forecast_prices = observation["price"].flatten()[
            :8
        ]  # use exactly the 8-step forecast
        avg_forecast_price = np.mean(forecast_prices)

        # Update short-term memory (length 8) with the latest forecast average
        self.price_memory.append(avg_forecast_price)
        # keep only the last 8 forecast means (2 h); the memory starts with 96 entries, so slicing is needed
        # (dropping one element per step, as before, kept it at 96 entries forever)
        self.price_memory = self.price_memory[-8:]

        # Compute current moving average from memory
        self.window = sum(self.price_memory) / len(self.price_memory)

        # Compare to moving average
        if avg_forecast_price > self.window * 1.1:
            return 1  # high price
        elif avg_forecast_price < self.window * 0.9:
            return -1  # low price
        else:
            return 0  # average price

    def choose_bes_actions(self, observation, price_state):
        """
        Decides the action of the battery based on the current price
        state and knowledge of the current SOC, and BES capacity.
        """
        actions = []
        soc = observation["soc_percent"][0]
        for i in range(self.storages_count):
            soc_percent = soc[i]
            max_e = self.max_e_mwh_bes[i]
            max_p = self.max_p_mw_bes[i]
            charge = max_e * soc_percent
            space = max_e - charge
            if price_state == 1:
                action = max(-charge, -max_p)
            elif price_state == 0:
                action = 0
            elif price_state == -1:
                action = min(space, max_p)
            action = action / max_p
            actions.append(action)

        return actions

    def set_env(self, env):
        """
        Sets the environment for the controller and extracts the maximum power
        and energy values for the different controllable devices.
        """

        self.max_p_mw_bes, self.max_e_mwh_bes = env.env_method(
            "get_controllable_storage_power"
        )[0]

        self.max_p_mw_ev, self.max_e_mwh_ev = env.env_method(
            "get_controllable_ev_power"
        )[0]

        controller_counts = env.env_method("get_controllers_count")[0]
        self.storages_count = controller_counts[0]
        self.pv_count = controller_counts[1]
        self.ev_count = controller_counts[2]
        self.hp_count = controller_counts[3]

    @staticmethod
    def scale_action(action):
        """
        Scales the action to the range [-1, 1]
        """
        return (action - 1) / (1 - (-1))
