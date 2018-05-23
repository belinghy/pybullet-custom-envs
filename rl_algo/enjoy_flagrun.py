import argparse
import time

import gym
import numpy as np
import pybullet as p
import pybullet_envs

import torch
from common import use_cuda
from ddpg import Actor


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--algo', default='DDPG',
                        help='algorithm to use: DDPG')
    parser.add_argument('--env-name', default="HumanoidFlagrunBulletEnv-v0",
                        help='name of the environment to run')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.999)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--ou_noise', type=bool, default=True)
    parser.add_argument('--param_noise', type=bool, default=False)
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    return parser.parse_args()

def main():
    args = parse_args()
    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = Actor(env.observation_space.shape[0], env.action_space)
    agent = agent.cuda() if use_cuda else agent

    # Load saved model
    agent.load_state_dict(torch.load('./models/ddpg_actor_HumanoidFlagrunBulletEnv-v0_'))

    total_numsteps = 0

    # Need this to render
    env.render("human")
    env.reset()
    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

    while True:
        state = torch.Tensor([env.reset()])
        state = state.cuda() if use_cuda else state

        score = 0
        frame = 0
        restart_delay = 0

        while True:
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            time.sleep(0.005)
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

            # Choose action
            action = agent(state).data
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            total_numsteps += 1
            score += reward
            frame += 1

            state = torch.Tensor([next_state])
            state = state.cuda() if use_cuda else state

            if not done:
                continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60 * 2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0:
                    break

    env.close()
    return


if __name__ == '__main__':
    main()
