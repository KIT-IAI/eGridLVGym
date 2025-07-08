import gymnasium as gym

"""
This file contains a wrapper that sets the seed of the
environment to the same value every time it is reset
This was needed because of the way the SLURM jobs are
spawning new processes, may not be needed on other systems.
"""


class SeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed

    def reset(self, seed=None):
        if seed is None:
            return self.env.reset(seed=self._seed)
        return self.env.reset(seed=seed)


def get_seeded_env_fn(env_id: str, seed: int):
    def fn(**kwargs):
        env = gym.make(env_id, **kwargs)
        env = SeedWrapper(env=env, seed=seed)
        return env

    return fn


def get_seeded_mo_env_fn(env_id: str, seed: int):
    import mo_gymnasium as mo_gym

    def fn(**kwargs):
        env = mo_gym.make(env_id, **kwargs)
        env = SeedWrapper(env=env, seed=seed)
        return env

    return fn
