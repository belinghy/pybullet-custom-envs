## Custom Environments

Installation of PyBullet includes also modules `pybullet_envs` and `pybullet_data`,
which contains implementations of some of the Gym Mujoco environments.  This directory contains the minimum from each to run simple Crab2D and Walker2D environments.

Somethings were modified using code from the newest PyBullet branch, and some code were further fixed (camera code).  There are likely other bugs that needs to be fixed.

To use the Crab2D and Walker2D,
```python
import gym
import cust_envs  # Will register environments with gym

env = gym.make("Crab2DCustomEnv-v0")  # or Walker2DCustomEnv-v0
```  