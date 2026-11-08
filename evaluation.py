"""
evaluation.py
This script evaluates a trained model on the evaluation episodes of the environment.
The number of parallel environments needs to match the number of environments the model was trained on.
"""

import sys
import os
import logging
from datetime import datetime

# import egridlv_gym
import pandas as pd
import argparse
import numpy as np

from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from sb3_contrib import TQC, TRPO, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from rule_based_controller import (
    RuleBasedController,
)
from utils.seed_wrapper import get_seeded_env_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
start_time = datetime.now()
logger.info(f"Starting evaluation script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")


def get_battery_p_action_indices(env_name, seed):
    # Dummy instance with seed (without VecEnv)
    dummy_env = get_seeded_env_fn(env_name, seed)()
    storages, _, _, _ = dummy_env.get_controllers_count()
    dummy_env.close()
    return list(range(storages))


def run_experiment(
    algorithm,
    num_envs=4,
    normalize=True,
    usecase="household-economic",
    obs_space="box",
    seed=58,
    enable_shielding=False,
    folder="./models/feeder-combined/best_trpo/best_model.zip",
    enforce_hard_grid_constraint=False,
):
    if folder is None:
        folder = f"./models/{usecase}/{algorithm}"

    if enable_shielding:
        logging.info("[INFO] VDE Shielding enabled")
        if enforce_hard_grid_constraint:
            logging.info("[INFO] Shielding and transformer hard constraint enforced.")
            output_path = f"./results/{usecase}/evaluations/hard_vde-shielding/eval_{algorithm}.csv"
        else:
            output_path = (
                f"./results/{usecase}/evaluations/vde-shielding/eval_{algorithm}.csv"
            )
    else:
        if enforce_hard_grid_constraint:
            logging.info("[INFO] Transformer hard constraint enforced.")
            output_path = f"./results/{usecase}/evaluations/hard/eval_{algorithm}.csv"
        else:
            output_path = f"./results/{usecase}/evaluations/eval_{algorithm}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Output path: {output_path}")

    rule_based_voltage = False
    recurrent = False
    logger.info(f"Loading model from: {folder}")

    # Load the corresponding RL or rule-based model
    if algorithm == "ppo":
        model = PPO.load(folder, verbose=1)
    elif algorithm == "ppo_tuned":
        model = PPO.load(folder, verbose=1)
    elif algorithm == "sac":
        model = SAC.load(folder, verbose=1)
    elif algorithm == "sac_curtailment_scaled":
        model = SAC.load(folder, verbose=1)
    elif algorithm == "ddpg":
        model = DDPG.load(folder, verbose=1)
    elif algorithm == "td3":
        model = TD3.load(folder, verbose=1)
    elif algorithm == "td3_curtailment_scaled":
        model = TD3.load(folder, verbose=1)
    elif algorithm == "tqc":
        model = TQC.load(folder, verbose=1)
    elif algorithm == "tqc_curtailment_scaled":
        model = TQC.load(folder, verbose=1)
    elif algorithm == "a2c":
        model = A2C.load(folder, verbose=1)
    elif algorithm == "trpo":
        model = TRPO.load(folder, verbose=1)
    elif algorithm == "recurrent_ppo":
        model = RecurrentPPO.load(folder, verbose=1)
        recurrent = True
    elif algorithm == "baseline":
        model = RuleBasedController(mode="baseline")
    elif algorithm == "baseline_price_window":
        model = RuleBasedController(mode="price_window")
    elif algorithm == "baseline_voltage":
        model = RuleBasedController(mode="baseline")
        rule_based_voltage = True
    else:
        raise ValueError("Invalid algorithm")

    # Select the appropriate environment
    if usecase == "household-economic":
        env_name = "eGridLV-household-economic-v0"
    elif usecase == "feeder-economic":
        env_name = "eGridLV-feeder-economic-v0"
    elif usecase == "feeder-voltage":
        env_name = "eGridLV-feeder-voltage-v0"
    elif usecase == "feeder-combined":
        env_name = "eGridLV-feeder-combined-v0"
    else:
        raise ValueError("Invalid usecase")

    # Initialize parallel environments
    env = make_vec_env(
        get_seeded_env_fn(env_name, seed),
        n_envs=num_envs,
        env_kwargs={
            "num_days": 7,
            "render_mode": None,
            "normalize": normalize,
            "observation_space": obs_space,
            "training": False,
            "rule_based_voltage": rule_based_voltage,
            "enable_shielding": enable_shielding,
            "enforce_trafo_limit": enforce_hard_grid_constraint,
            "transformer_threshold": 84.5,
        },
        vec_env_cls=SubprocVecEnv,
    )

    logger.info("*** evaluation.py started ***")
    logger.info(f"usecase: {usecase}")
    logger.info(f"algorithm: {algorithm}")
    logger.info(f"model path: {folder}")
    logger.info(f"seed: {seed}")
    logger.info(f"enable_shielding: {enable_shielding}")
    logger.info(f"enforce_hard_grid_constraint: {enforce_hard_grid_constraint}")

    # Setup logging path for environment metrics
    log_path = f"./results/{usecase}/final_eval_env_logs/"
    log_path += "vde-shielding/" if enable_shielding else ""
    log_path += algorithm
    env = VecMonitor(env, filename=log_path)
    logger.info(f"Logging VecMonitor to: {log_path}")

    model.set_env(env)
    global_result = []
    observation = env.reset()
    episode_starts = np.ones(num_envs, dtype=bool)
    lstm_states = None

    # Evaluate model
    for iteration in range(11):
        logger.info(f"Episode {iteration}/11")
        result = []
        cumulative_reward = 0
        truncated = False
        #        observation = env.reset(seed=seed)
        observation = env.reset()

        lstm_states = None
        episode_starts = np.ones(num_envs, dtype=bool)

        while not truncated:
            if recurrent:
                action, lstm_states = model.predict(
                    observation,
                    deterministic=True,
                    state=lstm_states,
                    episode_start=episode_starts,
                )
            else:
                action, _states = model.predict(observation, deterministic=True)
            obs, rewards, done, info = env.step(action)

            # Record episode results
            episode_starts = done
            info = info[0]
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

            cumulative_reward += rewards.flat[0]
            # Map actions for logging based on usecase
            actions = {}
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
            elif usecase == "feeder-combined":
                actions = {}
            if True in done:
                truncated = True
            # Append results to the global result list
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
    logger.info(f"Finished episode {iteration}")
    # Save evaluation result to CSV
    eval_episode_data = pd.DataFrame(global_result)
    # set the output path for the evaluation results
    if enable_shielding:
        if enforce_hard_grid_constraint:
            logging.info("[INFO] Shielding and transformer hard constraint enforced.")
            output_path = f"./results/{usecase}/evaluations/vde-shielding/hard_eval_{algorithm}.csv"
        else:
            logging.info("[INFO] VDE Shielding enabled")
            output_path = (
                f"./results/{usecase}/evaluations/vde-shielding/eval_{algorithm}.csv"
            )
    else:
        if enforce_hard_grid_constraint:
            logging.info("[INFO] Transformer hard constraint enforced.")
            output_path = f"./results/{usecase}/evaluations/hard_eval_{algorithm}.csv"
        else:
            output_path = f"./results/{usecase}/evaluations/eval_{algorithm}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the DataFrame to a CSV file
    eval_episode_data.to_csv(output_path)
    logging.info(
        f"Evaluation results saved to: {' with vde-shielding' if enable_shielding else ''} saved to: {output_path}"
    )
    logging.info(
        f"[FLAG] enable_shielding: {enable_shielding} | enforce_hard_grid_constraint: {enforce_hard_grid_constraint}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script.")
    # Add arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        help="RL algorithm to evaluate (from stable_baselines3, sb3_contrib, or rule-based control).",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments to run. Must match the training setup.",
    )
    parser.add_argument(
        "--no-normalization",
        default=False,
        action="store_true",
        help="Disable environment normalization (useful for rule-based controllers).",
    )
    parser.add_argument(
        "--usecase",
        type=str,
        default="feeder-combined",
        help="The use case to evaluate.",
    )
    parser.add_argument(
        "--obs-space",
        type=str,
        default="box",
        help="Observation space: 'box' (default) or 'dict' (e.g., for RBC).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="./models/feeder-combined/best_a2c/best_model",
        help="Optional path to model folder.",
    )
    parser.add_argument(
        "--seed", type=int, default=58, help="Random seed for environment."
    )
    parser.add_argument(
        "--enable_shielding",
        default=False,
        action="store_true",
        help="Enable reactive power control (shielding).",
    )
    parser.add_argument(
        "--enforce_hard_grid_constraint",
        default=False,
        action="store_true",
        help="Enforce critical grid constraints (e.g., transformer load).",
    )

    # Parse arguments
    args = parser.parse_args()
    # Log all arguments in a clean and structured way
    logger.info("Evaluation configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    run_experiment(
        algorithm=args.algorithm,
        num_envs=args.envs,
        normalize=not args.no_normalization,
        usecase=args.usecase,
        obs_space=args.obs_space,
        seed=args.seed,
        enable_shielding=args.enable_shielding,
        folder=args.folder,
        enforce_hard_grid_constraint=args.enforce_hard_grid_constraint,
    )
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Evaluation ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")
