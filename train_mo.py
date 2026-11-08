import sys
import numpy as np
import os
import argparse
import psutil
import logging
import wandb

# import egridlv_gym
import mo_gymnasium as mo_gym
import egridlv_gym  # noqa: F401  (registers the eGridLV environments)
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.capql.capql import CAPQL

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug(f"sys.argv: {sys.argv}")


def log_memory_usage(prefix=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024**2)
    logger.info(f"{prefix}[MEMORY] RSS Usage: {mem_mb:.2f} MB")


def get_env_and_ref_point(env_name, ref_point=None):
    if env_name == "household-economic":
        default_ref = np.array([4, -2])
        env_id = "mo-eGridLV-household-economic-v0"
    elif env_name == "feeder-voltage":
        default_ref = np.array([-0.1, -0.01, 0])
        env_id = "mo-eGridLV-feeder-voltage-v0"
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

    return env_id, ref_point if ref_point is not None else default_ref


def parse_str_to_array(s):
    """Converts a comma-separated string to a numpy array of floats."""
    return np.array([float(x.strip()) for x in s.split(",")])


def train_capql_agent(env_name, episodes=1000, wandb_log=False, ref_point=None, run_name=None):
    env_id, ref_point = get_env_and_ref_point(env_name, ref_point)
    save_dir = f"models/{run_name or env_name}"

    env = mo_gym.make(env_id)
    eval_env = mo_gym.make(env_id)
    env = MORecordEpisodeStatistics(env)
    best_reward = -np.inf

    agent = CAPQL(env, log=wandb_log, learning_starts=96, batch_size=96)
    timesteps_per_episode = 96 * 7
    total_timesteps = timesteps_per_episode * episodes
    # every 100 episodes
    eval_interval = timesteps_per_episode * 100

    logger.info("Starting training...")
    log_memory_usage("[Init] ")
    logger.info(f"Environment name: {env_name}")
    logger.info(f"Environment ID: {env_id}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Reference point: {ref_point}")
    # uniform weights for scalarization
    w_uni = np.ones_like(ref_point)

    for t in range(0, total_timesteps, eval_interval):
        agent.train(
            total_timesteps=eval_interval,
            ref_point=ref_point,
            eval_env=eval_env,
            eval_freq=total_timesteps,
        )

        # Evaluate performance
        obs, _ = eval_env.reset()
        done = False
        rewards = []

        while not done:
            action = agent.eval(obs, w=w_uni)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            done = terminated or truncated
            # scalarization sum over vector reward
            rewards.append(np.sum(reward))

        total_reward = np.sum(rewards)
        logger.info(f"Eval @ step {t+eval_interval}: R = {total_reward:.2f}")
        log_memory_usage(f"[Eval done @ step {t+eval_interval}] ")
        if total_reward > best_reward:
            filename = "best_capql/best_model"
            os.makedirs(os.path.join(save_dir, "best_capql"), exist_ok=True)
            agent.save(save_dir=save_dir, filename=filename)
            logger.info("New best model found!")
            logger.info(f"Saving to {os.path.join(save_dir, filename)}.tar")
            best_reward = total_reward
    filename = f"capql_final_{episodes}"
    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir=save_dir, filename=filename)
    logger.info("Training completed.")
    logger.info(f"Model saved to: {os.path.join(save_dir, filename)}.tar")


if __name__ == "__main__":
    """Parse CLI arguments and run training."""

    parser = argparse.ArgumentParser(description="Train CAPQL agent.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="household-economic",
        help="Env: 'household-economic' or 'feeder-voltage'",
    )
    parser.add_argument(
        "--episodes", type=int, default=8000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--ref_point",
        type=str,
        default=None,
        help="Custom reference point, e.g., '4,-2,2'",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Folder name under models/ (default: the environment name), e.g. feeder-voltage-capql-fix",
    )
    parser.add_argument(
        "--wandb_log",
        action="store_true",
        help="Enable logging to wandb. Default: False",
    )
    args = parser.parse_args()

    # Convert ref_point string to array if provided
    ref_point = parse_str_to_array(args.ref_point) if args.ref_point else None

    train_capql_agent(
        env_name=args.env_name,
        episodes=args.episodes,
        ref_point=ref_point,
        wandb_log=args.wandb_log,
        run_name=args.run_name,
    )
    logger.info("Training completed.")
