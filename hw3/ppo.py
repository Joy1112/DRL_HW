import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from distributions import Categorical, DiagGaussian
from replay_buffer import ReplayBuffer


class Policy(nn.Module):
    def __init__(self,
                 action_space,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None):
        super(Policy, self).__init__()

        self.action_space = action_space
        self.input_type = input_type
        self.input_feature = input_feature
        self.input_img_size = input_img_size

        if self.input_type == 'vector':
            self.actor = nn.Sequential(nn.Linear(input_feature, 128),
                                       nn.Tanh(),
                                       nn.Linear(128, 64),
                                       nn.Tanh())
            self.critic = nn.Sequential(nn.Linear(input_feature, 128),
                                        nn.Tanh(),
                                        nn.Linear(128, 64),
                                        nn.Tanh(),
                                        nn.Linear(64, 1))
        elif self.input_type == 'image':
            pass

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)

    def forward(self, x):
        raise NotImplementedError

    def act(self, obs, deterministic=False):
        value = self.critic(obs)
        actor_features = self.actor(obs)
        action_dist = self.dist(actor_features)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()

        action_log_probs = action_dist.log_probs(action)

        return value, action.item(), action_log_probs

    def getValue(self, obs):
        return self.critic(obs)

    def evalActions(self, obs, action):
        value = self.critic(obs)
        actor_features = self.actor(obs)
        action_dist = self.dist(actor_features)

        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class PPO():
    def __init__(self,
                 action_space,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 learning_rate,
                 update_times,
                 gamma=0.98,
                 lmbda=0.95,
                 input_type='vector',
                 input_feature=None,
                 input_img_size=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 device='cpu'):
        self.device = device
        self.actor_critic = Policy(action_space, input_type, input_feature, input_img_size).to(self.device)

        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.update_times = update_times

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        self.memory = []

    def insert(self, transition):
        self.memory.append(transition)

    def _makeBatch(self):
        state_lst, action_lst, reward_lst, next_state_lst, prob_a_lst, done_mask_lst = [], [], [], [], [], []
        for transition in self.memory:
            s, a, r, next_s, prob_a, done_mask = transition
            state_lst.append(s)
            action_lst.append([a])
            reward_lst.append([r])
            next_state_lst.append(next_s)
            prob_a_lst.append([prob_a])
            done_mask_lst.append([done_mask])
        self.memory = []

        return torch.tensor(state_lst, dtype=torch.float).to(self.device), \
               torch.tensor(action_lst).to(self.device), \
               torch.tensor(reward_lst).to(self.device), \
               torch.tensor(next_state_lst, dtype=torch.float).to(self.device), \
               torch.tensor(done_mask_lst, dtype=torch.float).to(self.device), \
               torch.tensor(prob_a_lst).to(self.device)

    def learn(self):
        s, a, r, next_s, done_mask, old_action_log_probs = self._makeBatch()

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for i in range(self.update_times):
            td_target = r + self.gamma * self.actor_critic.getValue(next_s) * done_mask
            delta = td_target - self.actor_critic.getValue(s)
            delta = delta.detach().cpu().numpy()

            adv_lst = []
            adv = 0.0
            for delta_t in delta[::-1]:
                adv = self.gamma * self.lmbda * adv + delta_t[0]
                adv_lst.append([adv])
            adv_lst.reverse()
            advantage = torch.tensor(adv_lst, dtype=torch.float).to(self.device)

            values, action_log_probs, dist_entropy = self.actor_critic.evalActions(s, a)
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(values, td_target.detach()).mean()
            loss = self.value_loss_coef * value_loss + action_loss - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

        num_updates = self.update_times * len(a)

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def sampleAction(self, obs, deterministic=False):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        value, action, action_log_probs = self.actor_critic.act(obs.to(self.device), deterministic)

        return action, action_log_probs
