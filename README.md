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
* Blending requires adjust parameters on the for each input state, 
no quick way to do this without using a loop for a mini-batch?  Changed batch size to `32`.
* Starting with two experts both with pre-trained weights.  One is fixed in evaluation mode, 
the other is set fro training.  Initially blending module is unbiased, 
but slowly we should see blending module to converge to favour first expert.
* Model trained for `160` episodes. Blending coefficients follow expected trend, 
but the bias is still balanced `.48` to `.52`.
* For deeper nets, there are more parameters to copy, which is time consuming. For smaller nets,
most of the work is being done on the CPU, which is slow. How to make this architecture run fast?