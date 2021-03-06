import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer


class Qnet(nn.Module):
    def __init__(self,
                 num_actions,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None,
                 dueling=False):
        super(Qnet, self).__init__()
        self.num_actions = num_actions
        self.input_type = input_type
        self.input_feature = input_feature
        self.input_img_size = input_img_size
        self.dueling = dueling

        self.initLayers()

    def initLayers(self):
        if self.input_type == 'vector':
            if self.dueling:
                self.fc_adv = nn.Sequential(
                    nn.Linear(self.input_feature, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.num_actions))
                self.fc_val = nn.Sequential(
                    nn.Linear(self.input_feature, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1))
            else:
                self.fc = nn.Sequential(
                    nn.Linear(self.input_feature, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.num_actions))
        elif self.input_type == 'image':
            pass

    def forward(self, x):
        if self.input_type == 'vector':
            if self.dueling:
                adv = self.fc_adv(x)
                val = self.fc_val(x)
                return val + (adv - adv.mean(dim=1, keepdim=True))
            else:
                return self.fc(x)
        elif self.input_type == 'image':
            return None


class DQN():
    def __init__(self,
                 num_actions,
                 gamma,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 update_times,
                 target_q_update_freq,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None,
                 prioritized=False,
                 random_action=False,
                 device='cpu'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_times = update_times
        self.target_q_update_freq = target_q_update_freq

        self.input_type = input_type
        self.input_feature = input_feature
        self.input_img_size = input_img_size
        self.prioritized = prioritized
        self.random_action = random_action
        self.device = device

        self.num_params_update = 0

        self.create_nets()

        self.memory = ReplayBuffer(self.buffer_size)

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

    def create_nets(self):
        self.q = Qnet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.q_target = Qnet(self.num_actions, self.input_type, self.input_feature, self.input_img_size).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

    def sampleAction(self, obs, epsilon=None):
        if self.random_action:
            assert epsilon is not None

        obs = torch.from_numpy(obs).float().unsqueeze(0)
        q_out = self.q(obs.to(self.device))
        random_digit = random.random()
        action = q_out.argmax().item() if random_digit >= epsilon else random.randint(0, self.num_actions - 1)
        return action

    def calc_loss(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch = self.memory.sample(self.batch_size)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_mask_batch = done_mask_batch.to(self.device)

        q_out = self.q(state_batch)
        q_values = q_out.gather(1, action_batch)
        next_s_max_q = self.q_target(next_state_batch).detach().max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + self.gamma * next_s_max_q * done_mask_batch
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        return loss

    def learn(self):
        episode_loss = 0.0
        for i in range(self.update_times):
            loss = self.calc_loss()
            episode_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.num_params_update += 1

            if self.num_params_update % self.target_q_update_freq == 0:
                self.q_target.load_state_dict(self.q.state_dict())

        return episode_loss / self.update_times


class DoubleDQN(DQN):
    def __init__(self,
                 num_actions,
                 gamma,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 update_times,
                 target_q_update_freq,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None,
                 prioritized=False,
                 random_action=False,
                 device='cpu'):
        DQN.__init__(self, num_actions, gamma, buffer_size, batch_size, learning_rate,
                     update_times, target_q_update_freq, input_type, input_feature,
                     input_img_size, prioritized, random_action, device)

    def calc_loss(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch = self.memory.sample(self.batch_size)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_mask_batch = done_mask_batch.to(self.device)

        q_out = self.q(state_batch)
        q_values = q_out.gather(1, action_batch)
        next_s_actions = self.q(next_state_batch).detach().argmax(axis=1)
        next_s_max_q = self.q_target(next_state_batch).gather(1, next_s_actions.unsqueeze(-1))
        expected_q_values = reward_batch + self.gamma * next_s_max_q * done_mask_batch
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        return loss


class DuelingDQN(DQN):
    def __init__(self,
                 num_actions,
                 gamma,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 update_times,
                 target_q_update_freq,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None,
                 prioritized=False,
                 random_action=False,
                 device='cpu'):
        DQN.__init__(self, num_actions, gamma, buffer_size, batch_size, learning_rate,
                     update_times, target_q_update_freq, input_type, input_feature,
                     input_img_size, prioritized, random_action, device)

    def create_nets(self):
        self.q = Qnet(self.num_actions, self.input_type, self.input_feature, self.input_img_size, dueling=True).to(self.device)
        self.q_target = Qnet(self.num_actions, self.input_type, self.input_feature, self.input_img_size, dueling=True).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
