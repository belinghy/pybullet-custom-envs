# Random Experiments

## Blending PyBullet Experiment

```
python flagrun.py
tensorboard --logdir runs
```

### 2018-05-20
Commit: `013feab0195c0c82c0b249106c3be65f0ccf5f13`
* Environment: `HumanoidFlagrunBulletEnv-v0`
* Disabled actor learning to let critic converge first, for `540` episodes

### 2018-05-21
Commit: `88c853169b11fea58cd106f14d902644ac601737`
* *Bug in implementation*

### 2018-05-22
Commit: `b8e649d636ea01277bd74c1ab90a89a20d64a36c`
* Blending of expert module can be done be scaling the input to non-linear layers, since non-linear layers don't have weights and scaling weights of linear layers is the same as scaling their outputs?
* Testing with `4` experts; blending module seems to always converge to favour one and not stay in between. No sinusoidal behaviour as observed in the mode-adaptive neural network paper; perhaps needing some kind of *history* in the input. 

Commit: `0cff1cc0ba8819be9d0140b9ad197c57728d6e75`
* Changing activation to `tanh` since the action space `limit` is `1` and `-1`. However, this does not seem to make a difference.
* Perhaps need to train longer.  The network is also relatively large at 256, 128 in hidden layers; where 64 is common.

### 2018-05-23
Commit: `b00aa9579e1c32083954e2690cba4d0e2d9c68b2`
* Added `Crab2DCustomEnv-v0` which is just modified from Walker2D; still need `pybullet_envs`, but shouldn't need `pybullet_data`.
* Still need to figure out how torque is scaled and what the observation space contains, especially if it contains foot contact by default.