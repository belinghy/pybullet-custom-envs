## Getting Started

To use the Crab2D and Walker2D,
```python
import gym
import cust_envs  # Will register environments with gym

env = gym.make("Crab2DCustomEnv-v0")  # or Walker2DCustomEnv-v0
```  

Could be useful for debugging,
```python
bullet_client = env.unwrapped._p
bullet_client.setGravity(0, 0, -9.8)  # Needs to be called after every reset

robot_id = -1
for i in range(bullet_client.getNumBodies()):
    if bullet_client.getBodyInfo(i)[1].decode() == 'walker2d':
        robot_id = i

for i in range(bullet_client.getNumJoints(robot_id)):
    print('joint', bullet_client.getJointInfo(robot_id, i))
    print('link', bullet_client.getLinkState(robot_id, i))

```