from gymnasium.envs.registration import register

register(
    id            = "RLCarla-v0",
    entry_point   = "rlcarla.envs.carla_env:CarlaEnv",
    max_episode_steps = 1500,
)
