## Custom Environments

Installation of PyBullet includes also modules `pybullet_envs` and `pybullet_data`,
which contains implementations of some of the Gym Mujoco environments.  This directory contains the minimum from each to run simple Crab2D and Walker2D environments. PyBullet doc [here](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3).

Somethings were modified using code from the newest PyBullet branch, and some code were further fixed (camera code).  There are likely other bugs that needs to be fixed.

Some things to be mindful of: contact detection via conaffinity, condim, and contype, size, density, and mass. Full mujoco documentation at: http://www.mujoco.org/book/modeling.html.

To use the Crab2D and Walker2D,
```python
import gym
import cust_envs  # Will register environments with gym

env = gym.make("Crab2DCustomEnv-v0")  # or Walker2DCustomEnv-v0
```  

Could be useful for debugging,
```python
robot_id = -1
for i in range(bullet_client.getNumBodies()):
    if bullet_client.getBodyInfo(i)[1].decode() == 'walker2d':
        robot_id = i

print(bullet_client.getDynamicsInfo(robot_id, -1))
print(bullet_client.getLinkState(robot_id, link_id))
```

### 2018-05-23
Commit: `b00aa9579e1c32083954e2690cba4d0e2d9c68b2`
* Added `Crab2DCustomEnv-v0` which is just modified from Walker2D; still need `pybullet_envs`, but shouldn't need `pybullet_data`.
* Still need to figure out how torque is scaled and what the observation space contains, especially if it contains foot contact by default.

### 2018-05-25
Commit: `435581386ae0b6d6e62be692a1d7f32da8f9ebde`
* Fixed some camera control issue
* Added simple PD controller test for Crab2D environment