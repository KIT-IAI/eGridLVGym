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
import morl_baselines.multi_policy.capql.capql as capql_module
from morl_baselines.common.evaluation import seed_everything

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
    elif env_name == "household-economic-pen":
        # third objective -2 x device penalties; ref point only used for the hypervolume log of CAPQL.train
        default_ref = np.array([-150, -50, -1500])
        env_id = "mo-eGridLV-household-economic-pen-v0"
    elif env_name == "feeder-voltage":
        default_ref = np.array([-0.1, -0.01, 0])
        env_id = "mo-eGridLV-feeder-voltage-v0"
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

    return env_id, ref_point if ref_point is not None else default_ref


def parse_str_to_array(s):
    """Converts a comma-separated string to a numpy array of floats."""
    return np.array([float(x.strip()) for x in s.split(",")])


def set_weight_angle(angle_deg):
    """Half-angle of the cone around the uniform weight from which CAPQL samples training weights.

    morl-baselines 1.1.0 hardcodes 22.5 deg inside CAPQL.train(); the sampler class is looked up in the module at call
    time, so replacing it there changes only the angle. Full coverage of the positive simplex needs 45 deg for two
    objectives; for three objectives 35.3 deg reaches the edge midpoints and 54.7 deg the corners.
    """
    base = capql_module.WeightSamplerAngle

    class FixedAngleSampler(base):
        def __init__(self, rwd_dim, angle, w=None):
            super().__init__(rwd_dim, np.pi * angle_deg / 180.0, w)

    capql_module.WeightSamplerAngle = FixedAngleSampler
    logger.info(f"CAPQL weight-sampling half-angle set to {angle_deg} deg (library default 22.5 deg)")


def train_capql_agent(env_name, episodes=1000, wandb_log=False, ref_point=None, run_name=None, seed=None, angle=None):
    env_id, ref_point = get_env_and_ref_point(env_name, ref_point)
    # model root, default ./models (e.g. EGRIDLV_MODELS_ROOT=models/run_a)
    save_dir = f"{os.environ.get('EGRIDLV_MODELS_ROOT', 'models')}/{run_name or env_name}"
    if angle is not None:
        set_weight_angle(angle)
    if seed is not None:
        seed_everything(seed)
        logger.info(f"Seed {seed}")

    env = mo_gym.make(env_id)
    eval_env = mo_gym.make(env_id)
    env = MORecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        eval_env.reset(seed=seed + 1000)
    best_reward = -np.inf

    agent = CAPQL(env, log=wandb_log, learning_starts=96, batch_size=96, seed=seed)
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
        help="Env: 'household-economic', 'household-economic-pen' (with penalty objective) or 'feeder-voltage'",
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
    parser.add_argument("--seed", type=int, default=None, help="Seed for numpy/torch/env (default: unseeded, as before)")
    parser.add_argument(
        "--angle", type=float, default=None,
        help="Half-angle in degrees of the CAPQL training-weight cone (default: library value 22.5)",
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
        seed=args.seed,
        angle=args.angle,
    )
    logger.info("Training completed.")
