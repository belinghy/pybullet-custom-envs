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

