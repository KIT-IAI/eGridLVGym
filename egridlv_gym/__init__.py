from gymnasium.envs.registration import register
import gymnasium

register(
    id="eGridLV-household-economic-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    kwargs={"grid": "toy", "use_case": "economic"},
)

register(
    id="eGridLV-feeder-economic-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    kwargs={"grid": "single_feeder", "use_case": "economic"},
)

register(
    id="eGridLV-feeder-voltage-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    kwargs={"grid": "single_feeder", "use_case": "voltage"},
)

register(
    id="eGridLV-feeder-combined-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    kwargs={"use_case": "combined", "grid": "single_feeder"},
)

register(
    id="mo-eGridLV-feeder-combined-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    kwargs={"mo": True, "use_case": "combined", "grid": "single_feeder"},
)

register(
    id="mo-eGridLV-feeder-voltage-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    kwargs={"mo": True, "use_case": "voltage", "grid": "single_feeder"},
)

register(
    id="mo-eGridLV-household-economic-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    kwargs={"mo": True, "use_case": "economic", "grid": "household"},
)
