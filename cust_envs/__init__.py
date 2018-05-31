import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


register(
    id="Walker2DCustomEnv-v0",
    entry_point="cust_envs.envs:Walker2DCustomEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="Crab2DCustomEnv-v0",
    entry_point="cust_envs.envs:Crab2DCustomEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="PDCrab2DCustomEnv-v0",
    entry_point="cust_envs.envs:PDCrab2DCustomEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="NabiRosCustomEnv-v0",
    entry_point="cust_envs.envs:NabiRosCustomEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)
