import gym
import pybullet as p
import pybullet_envs

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from flagrun_weights import *
import time


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


ENV_NAME = 'HumanoidFlagrunHarderBulletEnv-v0'
env = gym.make(ENV_NAME)
env.render('human')


def relu(x):
    return np.maximum(x, 0)


class SmallReactivePolicy:
    "Simple multi-layer perceptron policy, no internal state"

    def __init__(self, observation_space, action_space):
        assert weights_dense1_w.shape == (observation_space.shape[0], 256)
        assert weights_dense2_w.shape == (256, 128)
        assert weights_final_w.shape == (128, action_space.shape[0])

    def act(self, ob):
        x = ob
        x = relu(np.dot(x, weights_dense1_w) + weights_dense1_b)
        x = relu(np.dot(x, weights_dense2_w) + weights_dense2_b)
        x = np.dot(x, weights_final_w) + weights_final_b
        return x


def main():
    env = gym.make("HumanoidFlagrunBulletEnv-v0")
    env.render(mode="human")
    pi = SmallReactivePolicy(env.observation_space, env.action_space)

    env.reset()
    torsoId = -1
    for i in range(p.getNumBodies()):
        print(p.getBodyInfo(i))
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i
            print("found humanoid torso")

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        while 1:
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            time.sleep(0.005)
            # print("frame=",frame)
            camInfo = p.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            distance = camInfo[10]
            yaw = camInfo[8]
            pitch = camInfo[9]
            targetPos = [0.95*curTargetPos[0]+0.05*humanPos[0],
                         0.95*curTargetPos[1]+0.05*humanPos[1], curTargetPos[2]]
            p.resetDebugVisualizerCamera(distance, yaw, pitch, targetPos)

            still_open = env.render("human")
            if still_open == False:
                return
            if not done:
                continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60*2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0:
                    break


if __name__ == "__main__":
    main()
