from gym.envs.registration import registry, register, make, spec

# ------------bullet-------------

register(
    id='InvertedPendulumSwingupCustomEnv-v0',
    entry_point='custom_envs.gym_pendulum_envs:InvertedPendulumSwingupBulletEnv',
    max_episode_steps=1000,
    reward_threshold=800.0,
)

register(
    id='ReacherCustomEnv-v0',
    entry_point='custom_envs.gym_manipulator_envs:ReacherBulletEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)

# register(
#	id='AtlasBulletEnv-v0',
#	entry_point='pybullet_envs.gym_locomotion_envs:AtlasBulletEnv',
#	max_episode_steps=1000
#	)


def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all()
              if spec.id.find('Bullet') >= 0]
    return btenvs
