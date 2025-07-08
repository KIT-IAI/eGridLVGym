import logging
import argparse
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# import egridlv_gym
# import mo_gymnasium as mo_gym
# from morl_baselines.common.evaluation import seed_everything
from morl_baselines.multi_policy.capql.capql import CAPQL
from utils.seed_wrapper import get_seeded_mo_env_fn


# Configure logging with timestamp and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Initial log entries
logging.info("Script execution started")
logging.info(f"Command-line arguments: {sys.argv}")


def weight_scaling_experiment(
    model_folder="best_capql",
    algorithm="CAPQL",
    # usecase="feeder-voltage",
    usecase="household-economic",
    seed=58,
    grid_type="combined",
):
    logging.info(f"Starting weight scaling: {algorithm}, {usecase}")
    model_path = f"./models/{usecase}/{model_folder}/{algorithm}.tar"
    model_path = f"./models/{usecase}/{model_folder}/best_model.tar"

    rule_based_voltage = False

    if usecase == "household-economic":
        env_name = "mo-eGridLV-household-economic-v0"
    elif usecase == "feeder-economic":
        env_name = "mo-eGridLV-feeder-economic-v0"
    elif usecase == "feeder-voltage":
        env_name = "mo-eGridLV-feeder-voltage-v0"
    elif usecase == "feeder-combined":
        env_name = "mo-eGridLV-feeder-v0"
    else:
        raise ValueError("Invalid usecase")

    env = get_seeded_mo_env_fn(env_name, seed)(
        render_mode=None, training=False, rule_based_voltage=rule_based_voltage
    )

    model = CAPQL(env, log=False, learning_starts=96, batch_size=96)
    model.load(path=model_path)
    observation, info = env.reset()

    # Generate weights based on grid_type
    if grid_type == "linear":
        weights_linear1 = np.array(
            [
                [round(c, 3), round(f, 4)]
                # Start 0.02
                for c in np.linspace(0.02, 1.0, 10)
                for f in np.linspace(0.02, 1.0, 10)
            ]
        )
        logging.info("Using LINEAR weight grid.")
        weights_combined = np.vstack((weights_linear1))
        weights = np.unique(weights_combined, axis=0)
        num_linear = len(weights_linear1)
    elif grid_type == "combined":
        weights_linear1 = np.array(
            [
                [round(c, 3), round(f, 4)]
                # Start 0.02
                for c in np.linspace(0.02, 1.0, 10)
                for f in np.linspace(0.02, 1.0, 10)
            ]
        )
        log_values = np.logspace(-2, 1, 8)
        weights_log = np.array(
            [[round(c, 3), round(f, 4)] for c in log_values for f in log_values]
        )
        weights_combined = np.vstack((weights_linear1, weights_log))
        weights = np.unique(weights_combined, axis=0)
        logging.info("Using COMBINED (linear + logarithmic) weight grid.")
        num_linear = len(weights_linear1)
        num_log = len(weights_log)
        num_unique = len(weights)
    else:
        raise ValueError("Invalid grid type")

    # Log grid summary
    logging.info(f"Total weight combinations: {len(weights)}")

    if grid_type == "combined":
        logging.info("=== Weight Grid Summary ===")
        logging.info(f"Linear grid size         : {num_linear:>3} combinations")
        logging.info(f"Logarithmic grid size    : {num_log:>3} combinations")
        logging.info(f"Total unique combinations: {num_unique:>3} combinations")
        logging.info("===========================")
    elif grid_type == "linear":
        logging.info("=== Weight Grid Summary ===")
        logging.info(f"Linear grid size         : {num_linear:>3} combinations")
        logging.info("===========================")

    weights = weights.tolist()

    weights_results = []
    # for weight in weights:
    for idx, weight in enumerate(weights, 1):
        logging.info(f"Evaluating weight combination {idx}/{len(weights)}: {weight}")
        weight = np.array(weight)
        reward_cost = 0
        reward_flexibility = 0

        for iteration in range(11):
            truncated = False
            logging.info(f"Running iteration {iteration}")
            while not truncated:
                action = model.eval(observation, w=weight)
                observation, rewards, terminated, truncated, info = env.step(action)

                reward_cost = reward_cost + rewards[0]
                reward_flexibility = reward_flexibility + rewards[1]

                if truncated:
                    observation, info = env.reset()
        weights_results.append(
            {
                "weight_cost": weight[0],
                "weight_flexibility": weight[1],
                "reward_cost": reward_cost / 11,
                "reward_flexibility": reward_flexibility / 11,
            }
        )
        logging.debug(f"Current result: {weights_results[-1]}")
    final_results = pd.DataFrame(weights_results)
    final_results.to_csv(
        f"./results/{usecase}/evaluations/weights_scale/eval_{model_folder}_{grid_type}.csv"
    )
    logging.info(
        f"Results saved to: ./results/{usecase}/evaluations/weights_scale/eval_{model_folder}_{grid_type}.csv"
    )
    logging.info("Final weight scaling completed successfully.")


def run_experiment(
    model_folder="best_capql",
    algorithm="CAPQL",
    usecase="household-economic",
    weight=np.array([1, 1]),
    seed=58,
):
    model_path = f"./models/{usecase}/{model_folder}/best_model.tar"
    logging.info(f"Using model_path from: {model_path}")
    rule_based_voltage = False

    if usecase == "household-economic":
        env_name = "mo-eGridLV-household-economic-v0"
    elif usecase == "feeder-economic":
        env_name = "mo-eGridLV-feeder-economic-v0"
    elif usecase == "feeder-voltage":
        env_name = "mo-eGridLV-feeder-voltage-v0"
    elif usecase == "feeder-combined":
        env_name = "mo-eGridLV-feeder-combined-v0"
    else:
        raise ValueError("Invalid usecase")

    env = get_seeded_mo_env_fn(env_name, seed)(
        render_mode=None, training=False, rule_based_voltage=rule_based_voltage
    )

    model = CAPQL(env, log=False, learning_starts=96, batch_size=96)
    model.load(path=model_path)

    global_result = []
    observation, info = env.reset()

    for iteration in range(11):
        result = []
        cumulative_reward = 0
        reward_cost = 0
        reward_flexibility = 0
        reward_penalty = 0

        truncated = False
        logging.info(f"Running iteration {iteration}")
        while not truncated:
            action = model.eval(observation, w=weight)
            observation, rewards, terminated, truncated, info = env.step(action)
            flattened_info = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    for i in range(len(value)):
                        flattened_info[f"{key}_{i}"] = value[i]
                else:
                    flattened_info[key] = value

            for key, value in flattened_info.items():
                if isinstance(value, np.ndarray) and len(value) == 1:
                    flattened_info[key] = value[0]

            reward_cost = reward_cost + rewards[0]
            reward_flexibility = reward_flexibility + rewards[1]
            reward_penalty = (
                reward_penalty
                + info["battery_penalty"]
                + info["battery_penalty"]
                + info["charger_penalty"]
                + info["pv_penalty"]
                + info["hp_penalty"]
            )
            cumulative_reward = reward_cost + reward_flexibility + reward_penalty
            # cumulative_reward += sum(rewards)

            if usecase == "household-economic":
                actions = {
                    "action_bes_0": action.flat[0],
                    "action_ev_0": action.flat[1],
                    "action_hp_0": action.flat[2],
                }
            elif usecase == "feeder-economic":
                actions = {
                    "action_bes_0": action.flat[0],
                    "action_bes_1": action.flat[1],
                    "action_bes_2": action.flat[2],
                    "action_ev_0": action.flat[3],
                    "action_ev_1": action.flat[4],
                    "action_ev_2": action.flat[5],
                    "action_ev_3": action.flat[6],
                    "action_hp_0": action.flat[7],
                    "action_hp_1": action.flat[8],
                    "action_hp_2": action.flat[9],
                }
            elif usecase == "feeder-voltage":
                actions = {
                    "action_bes_0": action.flat[0],
                    "action_bes_1": action.flat[1],
                    "action_bes_2": action.flat[2],
                    "action_bes_3": action.flat[4],
                    "action_bes_4": action.flat[5],
                    "action_pv_0": action.flat[6],
                    "action_pv_1": action.flat[7],
                    "action_pv_2": action.flat[8],
                    "action_pv_3": action.flat[9],
                    "action_pv_4": action.flat[9],
                }

            result.append(
                {
                    **flattened_info,
                    **actions,
                    "reward": rewards.flat[0],
                    "cumulative_reward": cumulative_reward,
                    "seed": seed,
                    "iteration": iteration,
                }
            )
        global_result.extend(result)
        if truncated:
            observation, info = env.reset()
    eval_episode_data = pd.DataFrame(global_result)
    eval_episode_data.to_csv(f"./results/{usecase}/evaluations/eval_{model_folder}.csv")
    logging.info(
        f"Results saved to: ./results/{usecase}/evaluations/eval_{model_folder}.csv"
    )


if __name__ == "__main__":

    start_time = datetime.now()
    logging.info(f"Evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description="Run CAPQL evaluation experiments.")
    parser.add_argument(
        "--mode",
        choices=["weight_single", "weight_multi"],
        default="weight_single",
        help="Choose which experiment to run.",
    )
    parser.add_argument(
        "--model_folder", type=str, default="best_capql", help="Model folder name"
    )
    parser.add_argument("--algorithm", type=str, default="CAPQL", help="Algorithm name")
    parser.add_argument(
        "--usecase", type=str, default="household-economic", help="Use case to evaluate"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="fixed weights, e.g., '1,1'"
    )
    parser.add_argument(
        "--grid_type",
        choices=["linear", "combined"],
        default="combined",
        help="Type of weight grid to evaluate: 'linear' for surface plots, 'combined' (linear + log) for Pareto analysis",
    )

    parser.add_argument(
        "--seed", type=int, default=58, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.mode == "weight_single":
        if args.weights is None:
            print("Warning: No weights provided, using default [1, 1, 1]")
            if args.usecase == "household-economic":
                weight = np.array([1, 1])
            else:
                weight = np.array([1, 1, 1])
        else:
            try:
                weight = np.array([float(x) for x in args.weights.split(",")])
            except Exception:
                raise ValueError(
                    "Invalid format for --weights. Please use the format:\n"
                    '  python evaluation_mo.py --mode=weight_single --weights="4, -2, 2" --seed=42'
                )
        run_experiment(
            model_folder=args.model_folder,
            algorithm=args.algorithm,
            usecase=args.usecase,
            weight=weight,
            seed=args.seed,
        )
    elif args.mode == "weight_multi":
        weight_scaling_experiment(
            model_folder=args.model_folder,
            algorithm=args.algorithm,
            usecase=args.usecase,
            seed=args.seed,
            grid_type=args.grid_type,
        )
    end_time = datetime.now()
    elapsed = end_time - start_time
    logging.info(f"Evaluation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {str(elapsed)}")
