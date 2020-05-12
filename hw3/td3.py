import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ddpg import QNet, Actor, DDPG
from replay_buffer import ReplayBuffer


class TD3(DDPG):
    def __init__(self,
                 num_actions,
                 gamma,
                 tau,
                 buffer_size,
                 batch_size,
                 lr_critic,
                 lr_actor,
                 update_times,
                 update_actor_freq=3,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None,
                 device='cpu'):

        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.update_times = update_times
        self.update_actor_freq = update_actor_freq

        self.input_type = input_type
        self.input_feature = input_feature
        self.input_img_size = input_img_size
        self.device = device

        self.memory = ReplayBuffer(self.buffer_size)
        self.critic_update_times = 0

        self._createNets()
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

    def _createNets(self):
        self.critic_1 = QNet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.critic_1_target = QNet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = QNet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.critic_2_target = QNet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor = Actor(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.actor_target = Actor(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def _calcCriticLoss(self, state_batch, action_batch, reward_batch, next_state_batch):
        next_actions = self.actor_target(next_state_batch)
        next_target_values_1 = self.critic_1_target(next_state_batch, next_actions)
        next_target_values_2 = self.critic_2_target(next_state_batch, next_actions)
        next_target_values = torch.min(torch.cat((next_target_values_1, next_target_values_2), 1), dim=1)[0].unsqueeze(-1)
        target = reward_batch + self.gamma * next_target_values
        target = target.detach()

        action_batch_one_hot = F.one_hot(action_batch, num_classes=self.num_actions).squeeze(1)
        loss_1 = F.mse_loss(self.critic_1(state_batch, action_batch_one_hot), target)
        loss_2 = F.mse_loss(self.critic_2(state_batch, action_batch_one_hot), target)
        return loss_1, loss_2

    def _calcActorLoss(self, state_batch):
        loss = -self.critic_1(state_batch, self.actor(state_batch)).mean()
        return loss

    def learn(self):
        cumulated_critic_1_loss = 0.0
        cumulated_critic_2_loss = 0.0
        cumulated_actor_loss = 0.0
        actor_update_times = 0
        for i in range(self.update_times):
            s, a, r, next_s, done_mask = self._sampleMiniBatch()

            critic_1_loss, critic_2_loss = self._calcCriticLoss(s, a, r, next_s)
            cumulated_critic_1_loss += critic_1_loss
            cumulated_critic_2_loss += critic_2_loss
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            self._softUpdate(self.critic_1, self.critic_1_target)
            self._softUpdate(self.critic_2, self.critic_2_target)
            self.critic_update_times += 1

            if self.critic_update_times % self.update_actor_freq == 0:

                actor_loss = self._calcActorLoss(s)
                cumulated_actor_loss += actor_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self._softUpdate(self.actor, self.actor_target)
                actor_update_times += 1

        return cumulated_critic_1_loss / self.update_times, \
               cumulated_critic_2_loss / self.update_times, \
               cumulated_actor_loss / actor_update_times if actor_update_times > 0 else 0.0
