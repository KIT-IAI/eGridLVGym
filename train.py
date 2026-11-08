"""
train.py - Trains a single-objective RL agent on eGridLV environments.
The script sets up the selected use case, initializes the RL algorithm,
and trains the agent using SB3.
Results, logs, and models are saved for reproducibility and evaluation.
"""

import sys
import logging
from datetime import datetime
import argparse
import torch.nn as nn
import egridlv_gym
from stable_baselines3 import PPO, TD3, SAC, DDPG, A2C
from sb3_contrib import TQC, TRPO, RecurrentPPO

# from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# , DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from utils.seed_wrapper import get_seeded_env_fn

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug(f"sys.argv: {sys.argv}")


def select_model(algorithm, env):
    """
    Returns a preconfigured RL model based on the selected algorithm and environment.

    Parameters:
        algorithm (str): Name of the RL algorithm ('ppo', 'sac', 'td3', etc.).
        env (gym.Env): Environment for training.

    Returns:
        BaseAlgorithm: Initialized RL model instance.

    Raises:
        ValueError: If the algorithm name is invalid.
    """
    logger.debug(f"Selecting model for algorithm: {algorithm}")
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            stats_window_size=10,
            use_sde=True,
            batch_size=96,
            n_steps=96 * 7,
        )
    elif algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            stats_window_size=10,
            learning_starts=96,
            use_sde=True,
        )
    elif algorithm == "ddpg":
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            learning_starts=96,
        )
    elif algorithm == "td3":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            stats_window_size=10,
            learning_starts=96,
        )
    elif algorithm == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            use_sde=True,
            stats_window_size=10,
        )
    elif algorithm == "tqc":
        model = TQC(
            "MlpPolicy",
            env,
            verbose=1,
            stats_window_size=10,
            use_sde=True,
            learning_starts=96,
        )
    elif algorithm == "trpo":
        model = TRPO(
            "MlpPolicy",
            env,
            verbose=1,
            stats_window_size=10,
            use_sde=True,
            n_steps=96 * 7,
            batch_size=96,
        )
    elif algorithm == "recurrent_ppo":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            stats_window_size=10,
            n_steps=96,
            batch_size=96,
        )
    elif algorithm == "ppo_tuned":
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            log_std_init=-0.6746674797556187,
            activation_fn=nn.ReLU,
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            stats_window_size=10,
            use_sde=True,
            n_steps=32,
            batch_size=64,
            gamma=0.98,
            learning_rate=0.0023362940487925365,
            ent_coef=0.01065453864731708,
            clip_range=0.1,
            n_epochs=1,
            gae_lambda=0.95,
            max_grad_norm=0.9,
            vf_coef=0.2546142404500582,
            sde_sample_freq=8,
            policy_kwargs=policy_kwargs,
        )
    else:
        logger.error(f"Invalid algorithm name: {algorithm}")
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return model


if __name__ == "__main__":
    """Parse CLI arguments and run training for selected algorithms."""

    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "--algorithm", type=str, default="ppo", help="The algorithm to train"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="The number of episodes to train."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="The number of environments to train on in parallel.",
    )
    parser.add_argument(
        "--usecase",
        type=str,
        default="household-economic",
        help="The use case to train on.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for the environment."
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=500,
        help="The evaluation frequency (number of episodes).",
    )
    parser.add_argument(
        "--curtailment-scaling",
        type=float,
        default=1.0,
        help="The scaling factor for the curtailment reward.",
    )
    parser.add_argument(
        "--reward-variant",
        type=str,
        default="default",
        choices=["default", "qu", "band", "band_q", "l1_qu", "bowl_qu", "hinge_qu", "step_qu", "count_qu", "bowl_qu_l10", "l1_qu_l10", "default_l10"],
        help="Reward variant: default (original reward); combined use case: qu (dead-band voltage term, line term "
        "above 80 %, PV reactive power cost); voltage use case: band, band_q, qu, l1_qu, bowl_qu, hinge_qu, step_qu, "
        "count_qu and *_l10 (line term weighted by 10).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Folder name for models and logs (default: the use case), e.g. feeder-combined-pf-s42, so that runs "
        "with different settings do not overwrite each other.",
    )
    args = parser.parse_args()
    logger.debug(f"Parsed arguments: {args}")
    run_dir = args.run_name or args.usecase

    algorithm = args.algorithm
    time_steps = args.episodes * 96 * 7
    eval_freq = args.eval_freq * 96 * 7

    # Map usecase to environment name
    if args.usecase == "household-economic":
        env_name = "eGridLV-household-economic-v0"
    elif args.usecase == "feeder-economic":
        env_name = "eGridLV-feeder-economic-v0"
    elif args.usecase == "feeder-voltage":
        env_name = "eGridLV-feeder-voltage-v0"
    elif args.usecase == "feeder-combined":
        env_name = "eGridLV-feeder-combined-v0"
    else:
        logger.error(f"Invalid use case: {args.usecase}")
        raise ValueError(f"Invalid use case: {args.usecase}")

    # Training and env configuration
    logger.info(
        f"\nSelected Use Case     : {args.usecase} (Env: {env_name})\n"
        f"Training Configuration:\n"
        f"  Algorithm            : {algorithm}\n"
        f"  Episodes             : {args.episodes} (Total Timesteps: {time_steps})\n"
        f"  Evaluation Frequency : Every {args.eval_freq} episodes ({eval_freq} timesteps)\n"
        f"  Parallel Envs        : {args.num_envs}\n"
        f"  Curtailment Scaling  : {args.curtailment_scaling}\n"
        f"  Seed                 : {args.seed}\n"
    )

    # Start training
    start_time = datetime.now()
    logger.info(f"Training start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Training environments setup
    train_env = make_vec_env(
        env_name,
        n_envs=args.num_envs,
        env_kwargs={
            "curtailment_cost_scaling": args.curtailment_scaling,
            "render_mode": None,
            "reward_variant": args.reward_variant,
        },
        vec_env_cls=SubprocVecEnv,
        seed=args.seed,
    )

    # Distinguish model name if curtailment scaling != 1.0
    if args.curtailment_scaling != 1.0:
        algorithm_name = f"{algorithm}_curtailment_scaled"
    else:
        algorithm_name = algorithm

    # Wrap environment for logging/monitoring
    train_env = VecMonitor(
        train_env,
        filename=f"results/{run_dir}/train_env_logs/{args.num_envs}_{algorithm_name}_{args.seed}",
    )  # /train_{args.num_envs}_{algorithm_name}_{args.seed}

    # Select model based on algorithm
    model = select_model(algorithm, train_env)

    # Evaluation environments setup
    eval_env = make_vec_env(
        get_seeded_env_fn(env_name, seed=42),
        n_envs=args.num_envs,
        env_kwargs={
            "training": True,
            "render_mode": None,
            "curtailment_cost_scaling": args.curtailment_scaling,
            "reward_variant": args.reward_variant,
        },
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = VecMonitor(
        eval_env,
        filename=f"results/{run_dir}/eval_env_logs/{args.num_envs}_{algorithm_name}_{args.seed}",
    )

    # Evaluation callback setup
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{run_dir}/best_{algorithm_name}/",
        eval_freq=max(eval_freq // args.num_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=11 * args.num_envs,
        log_path=f"results/{run_dir}/eval_logs/{args.num_envs}_{algorithm_name}_{args.seed}",
    )

    logger.info(f"Starting '{algorithm}' training for {args.episodes} episodes.")

    # Training Model
    model.learn(
        total_timesteps=time_steps,
        progress_bar=False,
        log_interval=1,
        reset_num_timesteps=True,
        callback=eval_callback,
    )

    # Saving Model
    model_save_path = f"models/{run_dir}/{algorithm_name}"
    model.save(model_save_path)

    # Log training end time, duration, and save location
    end_time = datetime.now()
    elapsed = end_time - start_time

    logger.info(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training duration: {elapsed}")
    logger.info(f"Model saved to: {model_save_path}")
