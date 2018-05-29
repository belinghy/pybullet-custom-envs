import os
# Need to import cust_envs
current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import cust_envs
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CModule(nn.Module):
    """ In original paper, this network is just one layer trained with CMA-ES """
    def __init__(self, env, latent_dim, hidden_dim):
        super(CModule, self).__init__()
        self.env = env
        action_dim = env.action_space.shape[0]
        self.action_dim = action_dim
        # latent_dim is size of z, hidden_dim is size of h
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        h_size = 32
        self.fc1 = nn.Linear(latent_dim + hidden_dim, h_size)
        self.out = nn.Linear(h_size, action_dim)

    def forward(self, x):
        """ Input should be concat(z, h) """
        x = F.relu(self.fc1(x))
        x = F.tanh(self.out(x))
        return x


class VModule(nn.Module):
    """ In original paper, this is a VAE. """
    def __init__(self, env, latent_dim):
        super(VModule, self).__init__()
        self.env = env
        observation_dim = env.observation_space.shape[0]
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim

        h_size = 32
        self.fc1 = nn.Linear(observation_dim, h_size)
        self.mu = nn.Linear(h_size, latent_dim)
        self.logvar = nn.Linear(h_size, latent_dim)

        self.fc2 = nn.Linear(latent_dim, h_size)
        self.fc3 = nn.Linear(h_size, observation_dim)

    def encode(self, x):
        """ Input should be observation, o """
        h1 = F.relu(self.fc1(x))
        return self.mu(h1), self.logvar(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        """ z is vector in latent space """
        h2 = F.relu(self.fc2(z))
        # Observation space is unbound, no activation
        return self.fc3(h2)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def main():
    env_name = 'Crab2DCustomEnv-v0'
    env = gym.make(env_name)

    # Must render once to create the window
    env.render("human")
    obs = env.reset()

    # Algorithm constants
    random_seed = 16
    num_iterations = 100
    observation_dim = env.observation_space.shape[0]
    latent_dim = 8 if observation_dim // 2 > 8 else observation_dim // 2
    hidden_dim = 32

    # Init stuff
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    c_module = CModule(env, latent_dim, hidden_dim)
    v_module = VModule(env, latent_dim)

    for i_iter in range(num_iterations):
        # Collect episodes

        # Train V
        # Train M
        # Train C



if __name__ == "__main__":
    main()