import argparse
from tensorboardX import SummaryWriter

import gym
import numpy as np
from gym import wrappers
import pybullet as p
import pybullet_envs

import torch
from common import use_cuda
from ddpg import DDPG
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition


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
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
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

    writer = SummaryWriter()

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = DDPG(args.gamma, args.tau,
                     env.observation_space.shape[0], env.action_space)
    agent = agent.cuda() if use_cuda else agent

    # Hardcode model path, only load critic for now
    agent.load_model(None, './models/ddpg_critic_HumanoidFlagrunBulletEnv-v0_')

    memory = ReplayMemory(args.replay_size)

    ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                         desired_action_stddev=args.noise_scale,
                                         adaptation_coefficient=1.05) if args.param_noise else None

    rewards = []
    total_numsteps = 0
    updates = 0

    # Need this to render
    if args.gui:
        env.render("human")
        env.reset()
        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i

    for i_episode in range(args.num_episodes):
        state = torch.Tensor([env.reset()])

        if args.ou_noise:
            ounoise.scale = (args.noise_scale - args.final_noise_scale) * \
                            max(0, args.exploration_end - i_episode) / args.exploration_end + args.final_noise_scale
            ounoise.reset()

        if args.param_noise and args.algo == "DDPG":
            agent.perturb_actor_parameters(param_noise)

        episode_reward = 0
        while True:
            if args.gui:
                distance, yaw = 5, 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
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

            action = agent.select_action(state, ounoise, param_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            if len(memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))

                    value_loss, policy_loss = agent.update_parameters(batch)

                    writer.add_scalar('loss/value', value_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)

                    updates += 1
            if done:
                break

        writer.add_scalar('reward/train', episode_reward, i_episode)

        # Update param_noise based on distance metric
        if args.param_noise:
            episode_transitions = memory.memory[memory.position-t:memory.position]
            states = torch.cat([transition[0]
                                for transition in episode_transitions], 0)
            unperturbed_actions = agent.select_action(states, None, None)
            perturbed_actions = torch.cat(
                [transition[1] for transition in episode_transitions], 0)

            ddpg_dist = ddpg_distance_metric(
                perturbed_actions.numpy(), unperturbed_actions.numpy())
            param_noise.adapt(ddpg_dist)

        rewards.append(episode_reward)
        if i_episode % 10 == 0:
            state = torch.Tensor([env.reset()])
            f1, f2 = agent.return_coefficients(state)
            writer.add_scalar('coeffs/1', f1, i_episode)
            writer.add_scalar('coeffs/2', f2, i_episode)
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state
                if done:
                    break

            writer.add_scalar('reward/test', episode_reward, i_episode)

            rewards.append(episode_reward)
            print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(
                i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))

            agent.save_model(args.env_name)

    env.close()
    return


if __name__ == '__main__':
    main()
