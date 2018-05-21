import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from common import use_cuda
from flagrun_weights import *


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class BlendingModule(nn.Module):
    def __init__(self, num_inputs, num_experts):
        """ Input should have some kind of history, maybe state + prev_action"""
        super(BlendingModule, self).__init__()
        hidden_size = 128
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.ln2 == nn.Linear(hidden_size)
        self.omega = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.softmax(self.omega(x))


class PolicyModule(nn.Module):
    def __init__(self, num_inputs, action_space, pretrained=False):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        l1_size = 256
        l2_size = 128

        self.linear1 = nn.Linear(num_inputs, l1_size)
        # self.ln1 = nn.LayerNorm(l1_size)
        self.linear2 = nn.Linear(l1_size, l2_size)
        # self.ln2 = nn.LayerNorm(l2_size)
        self.mu = nn.Linear(l2_size, num_outputs)

        # Init
        if pretrained:
            self.linear1.weight.data = torch.from_numpy(weights_dense1_w.T)
            self.linear1.bias.data = torch.from_numpy(weights_dense1_b.T)
            self.linear2.weight.data = torch.from_numpy(weights_dense2_w.T)
            self.linear2.bias.data = torch.from_numpy(weights_dense2_b.T)
            self.mu.weight.data = torch.from_numpy(weights_final_w.T)
            self.mu.bias.data = torch.from_numpy(weights_final_b.T)


    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        # x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = self.ln2(x)
        x = F.relu(x)
        mu = F.tanh(self.mu(x))
        return mu


class Actor(nn.Module):
    def __init__(self, num_inputs, action_space, num_experts):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_actions = action_space.shape[0]

        # Policy 1 is pre-trained, set to eval
        self.p1 = PolicyModule(num_inputs, num_actions, pretrained=True)
        self.p1.eval()

        # Training Policy 2 as additional expert
        self.p2 = PolicyModule(num_inputs, num_actions)

        # Combined policy
        self.cp = PolicyModule(num_inputs, num_actions)

        # Hardcode num_experts for now
        num_experts = 2
        self.blending_module = BlendingModule(num_inputs+num_actions, num_experts)

    def forward(self, state, prev_action):
        x = torch.cat((state, prev_action), dim=1)
        coeffs = self.blending_module(x)

        p1_states = self.p1.state_dict()
        p2_states = self.p2.state_dict()
        for name, _ in self.cp.named_parameters():
            print(p1_states[name])
            print(p2_states[name])

        #self.cp.load_state_dict()

        mu = self.cp(state)
        return mu


class Critic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        hidden_size = 128
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size+num_outputs, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.linear1(x)
        # x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)
        x = self.linear2(x)
        # x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V


class DDPG(nn.Module):
    def __init__(self, gamma, tau, num_inputs, action_space):
        super(DDPG, self).__init__()

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(self.num_inputs, self.action_space)
        self.actor_target = Actor(self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(self.num_inputs, self.action_space)
        self.critic_target = Critic(self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None, param_noise=None):
        self.actor.eval()
        t_state = Variable(state).cuda() if use_cuda else Variable(state)
        if param_noise is not None: 
            mu = self.actor_perturbed(t_state)
        else:
            mu = self.actor(t_state)

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            t_noise = torch.Tensor(action_noise.noise())
            t_noise = t_noise.cuda() if use_cuda else t_noise
            mu += t_noise

        return mu.clamp(-1, 1)


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))

        if use_cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            mask_batch = mask_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        # self.actor_optim.step()

        # soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))