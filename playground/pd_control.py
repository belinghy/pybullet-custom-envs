import os

current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import argparse
import cust_envs
import gym
import numpy as np
import time


class PDController:

    def __init__(self, env):
        self.action_dim = env.action_space.shape[0]
        self.high = env.action_space.high
        self.low = env.action_space.low
        frequency = 2
        self.k_p = (2 * np.pi * frequency) ** 2
        damping_ratio = 1
        self.k_d = 2 * damping_ratio * 2 * np.pi * frequency

    def drive_torques(self, targets, states):
        # States and targets should be [theta, omega] * action_dim
        diff = targets - states
        torques = self.k_p * diff[0::2] + self.k_d * diff[1::2]
        return np.clip(torques, self.low, self.high)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple PD control for driving characters in environments"
    )
    parser.add_argument(
        "--env", default="Crab2DCustomEnv-v0", help="Name of Gym environment to run"
    )
    return parser.parse_args()


def compute_targets(env):
    action_dim = env.action_space.shape[0]

    # Everything is normalized
    if env.spec.id == "Crab2DCustomEnv-v0":
        targets = np.array(
            [
                [-1, -1, -1, -1, -1, -1],  # split,
                [0, 0, 0, 0, 0, 0],  # frog
                [1, -1, 0, 1, -1, 0],  # tall stance
            ]
        )
    elif env.spec.id == "Walker2DCustomEnv-v0":
        targets = np.array(
            [
                [1, 1, 0, -1, 1, 0],  # split
                [-1, 1, 0, 1, 1, 0],  # split
                [1, 1, 0, 1, 1, 0],  # tall stance
            ]
        )
    random_choice = np.random.choice(targets.shape[0], 1)

    target_thetas = targets[random_choice]
    target_omegas = np.zeros(action_dim)

    targets = np.empty((2 * action_dim), dtype=np.float32)
    targets[0::2] = target_thetas
    targets[1::2] = target_omegas
    return targets


def main():
    args = parse_args()
    env = gym.make(args.env)

    # Must render then reset at least once first?
    env.render("human")
    obs = env.reset()

    # Needs to be called after every reset
    bullet_client = env.unwrapped._p
    bullet_client.setGravity(0, 0, -9.8)

    # Init setup
    action_dim = env.action_space.shape[0]
    controller = PDController(env)
    targets = compute_targets(env)

    while True:
        # Extract angles and velocities
        thetas_and_omegas = obs[8 : 8 + 2 * action_dim]
        if np.linalg.norm(targets - thetas_and_omegas, 1) / np.size(targets) < 0.1:
            # Converged to target pose
            targets = compute_targets(env)
        action = controller.drive_torques(targets, thetas_and_omegas)

        time.sleep(1. / 60.)
        # Had to hack camera_adjust() in WalkerBaseBulletEnv
        # env.unwrapped.camera_adjust(distance=5)

        obs, r, done, _ = env.step(action)

        env.render("human")


if __name__ == "__main__":
    main()
