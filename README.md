## Getting Started

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