from gymnasium.envs.registration import register
import gymnasium

register(
    id="eGridLV-household-economic-v0",
    entry_point="egridlv_gym.envs:BaseElectricGym",
    max_episode_steps=96 * 7,
    # was grid="toy", which BaseElectricGym.create_grid does not know -> it built the whole single feeder (identical to
    # feeder-economic). Fixed 2026-11-17 to the single household, as in mo-eGridLV-household-economic-v0 (CAPQL).
    kwargs={"grid": "household", "use_case": "economic"},
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
