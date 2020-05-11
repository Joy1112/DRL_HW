import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from distributions import Categorical
from replay_buffer import ReplayBuffer


class OrnsteinUhlenbeckNoise():
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=0.01, x_0=None):
        """
        The noise which will be added to the action according to the DDPG paper.
        """
        self.mu, self.sigma, self.theta, self.dt = mu, sigma, theta, dt
        self.x_0 = x_0
        self.reset()

    def reset(self):
        self.x_prev = self.x_0 if self.x_0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class QNet(nn.Module):
    def __init__(self,
                 num_actions,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None):
        """
        Q Networks for DDPG. (Critic)
        """
        super(QNet, self).__init__()
        self.num_actions = num_actions
        self.input_type = input_type
        self.input_feature = input_feature
        self.input_img_size = input_img_size

        self.createLayers()

    def createLayers(self):
        """
        Create the networks.
        """
        if self.input_type == 'vector':  # take vector as input
            self.fc_s = nn.Sequential(nn.Linear(self.input_feature, 64),
                                      nn.ReLU())
            self.fc_a = nn.Sequential(nn.Linear(self.num_actions, 64),
                                      nn.ReLU())
            self.fc_q = nn.Sequential(nn.Linear(128, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 1))
        elif self.input_type == 'image':  # The case when input_type is image is not finished
            pass

    def forward(self, x, a):
        if self.input_type == 'vector':
            v1 = self.fc_s(x)
            v2 = self.fc_a(a.float())
            return self.fc_q(torch.cat([v1, v2], dim=1))
        elif self.input_type == 'image':
            return None


class Actor(nn.Module):
    def __init__(self,
                 num_outputs,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None):
        """
        Policy net for DDPG. Please note that the output is the actor feature vector.
        """
        super(Actor, self).__init__()
        self.num_outputs = num_outputs
        self.input_type = input_type
        self.input_feature = input_feature
        self.input_img_size = input_img_size

        self.createLayers()

    def createLayers(self):
        """
        Create the networks.
        """
        if self.input_type == 'vector':  # take vector as input
            self.fc_block = nn.Sequential(nn.Linear(self.input_feature, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, self.num_outputs))
        elif self.input_type == 'image':  # The case when input_type is image is not finished
            pass

    def forward(self, x):
        if self.input_type == 'vector':
            x = self.fc_block(x)
            x = F.gumbel_softmax(x, tau=1, hard=True)
            return x
        elif self.input_type == 'image':
            return None


# class Actor(nn.Module):
#     def __init__(self,
#                  num_outputs,
#                  input_type='vector',
#                  input_feature=None,
#                  input_img_size=None):
#         """
#         The actor for DDPG.
#         Note that here use the categorical distribution for the action.
#         """
#         super(Actor, self).__init__()
#         self.num_outputs = num_outputs
#         self.input_type = input_type
#         self.input_feature = input_feature
#         self.input_img_size = input_img_size

#         self.base = MuNet(num_outputs=self.num_outputs,
#                           input_type=self.input_type,
#                           input_feature=self.input_feature,
#                           input_img_size=self.input_img_size)

#         # self.dist = Categorical(self.base.num_outputs, self.num_outputs)

#     def forward(self, x, noise=None, deterministic=True):
#         actor_features = self.base(x)
#         if noise is not None:
#             actor_features += noise
#         # dist = self.dist(actor_features)

#         # if deterministic:
#         #     action = dist.mode()
#         # else:
#         #     action = dist.sample()
#         action = F.gumbel_softmax(actor_features, tau=1, hard=True)

#         return action


class DDPG():
    def __init__(self,
                 action_dim,
                 num_actions,
                 gamma,
                 tau,
                 buffer_size,
                 batch_size,
                 lr_critic,
                 lr_actor,
                 update_times,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None,
                 prioritized=False,
                 device='cpu'):
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.update_times = update_times            # when learn the network, the number of updates for the q network

        self.input_type = input_type
        self.input_feature = input_feature
        self.input_img_size = input_img_size
        self.prioritized = prioritized              # use for the prioritized replay buffer, but it has not been used yet.
        self.device = device

        self.memory = ReplayBuffer(self.buffer_size)

        self._createNets()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.actor.base.num_outputs))

    def _createNets(self):
        # self.critic = QNet(self.action_dim, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        # self.critic_target = QNet(self.action_dim, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.critic = QNet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.critic_target = QNet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.actor_target = Actor(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def _softUpdate(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def _sampleMiniBatch(self):
        """
        Sample a mini-batch from the replay buffer and move it to the device
        """
        state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch = self.memory.sample(self.batch_size)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_mask_batch = done_mask_batch.to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch

    def _calcCriticLoss(self, state_batch, action_batch, reward_batch, next_state_batch):
        # next_action_batch = self.actor_target(next_state_batch).argmax(axis=1).unsqueeze(-1)
        # target = reward_batch + self.gamma * self.critic_target(next_state_batch, next_action_batch)
        target = reward_batch + self.gamma * self.critic_target(next_state_batch, self.actor_target(next_state_batch))
        action_batch_one_hot = F.one_hot(action_batch, num_classes=self.num_actions).squeeze(1)
        loss = F.mse_loss(self.critic(state_batch, action_batch_one_hot), target.detach())
        return loss

    def _calcActorLoss(self, state_batch):
        # current_action_batch = self.actor(state_batch).argmax(axis=1).unsqueeze(-1)
        # loss = - self.critic(state_batch, current_action_batch).mean()
        loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        return loss

    def sampleAction(self, obs, epsilon=0.0):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        # noise = torch.from_numpy(self.ou_noise()).float() * epsilon
        action_one_hot = self.actor(obs.to(self.device))
        action = action_one_hot.argmax().item() if random.random() >= epsilon else random.randint(0, self.num_actions - 1)

        return action

    def learn(self):
        cumulated_critic_loss = 0.0
        cumulated_actor_loss = 0.0
        for i in range(self.update_times):
            s, a, r, next_s, done_mask = self._sampleMiniBatch()

            critic_loss = self._calcCriticLoss(s, a, r, next_s)
            cumulated_critic_loss += critic_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = self._calcActorLoss(s)
            cumulated_actor_loss += actor_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._softUpdate(self.critic, self.critic_target)
            self._softUpdate(self.actor, self.actor_target)

        return cumulated_critic_loss / self.update_times, cumulated_actor_loss / self.update_times
