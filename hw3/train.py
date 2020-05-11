import os
import numpy as np
import random
# import gym
import gfootball.env as football_env
from optparse import OptionParser
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import OrnsteinUhlenbeckNoise, DDPG
from ppo import PPO
from create_logger import create_logger, print_and_log, print_and_write


parser = OptionParser()
parser.add_option("-m", "--model", dest="model", type="str", help="The model/algorithm which to be used. (DDPG, TD3 or PPO)", default='DDPG')
parser.add_option("-e", "--env", dest="env", type="str", help="Which environment to run the experiment. (academy_3_vs_1_with_keeper or academy_empty_goal)", default='academy_3_vs_1_with_keeper')
parser.add_option("-g", "--gamma", dest="gamma", type="float", help="The discount factor", default=0.98)
parser.add_option("-t", "--tau", dest="tau", type="float", help="The factor for target network soft update", default=0.005)
parser.add_option("-b", "--buffer_size", dest="buffer_size", type="int", help="The size of the replay buffer", default=200000)
parser.add_option("--learning_rate", dest="learning_rate", type="float", help="The learning rate for PPO", default=0.0005)
parser.add_option("--clip_param", dest="clip_param", type="float", help="The clip parameter for PPO value loss", default=0.2)
parser.add_option("--device", dest="device", type="str", help="The device to run the experiment. (cuda or cpu)", default='cpu')
parser.add_option("--update_times", dest="update_times", type="int", help="The number of updates every episode", default=5)
parser.add_option("--max_episodes", dest="max_episodes", type="int", help="The maximum number of episodes", default=500000)

cfg, in_args = parser.parse_args()


def trainDDPG(env, writer, logger):
    # create model
    model = DDPG(action_dim=1,
                 num_actions=env.action_space.n,
                 gamma=cfg.gamma,
                 tau=cfg.tau,
                 buffer_size=cfg.buffer_size,
                 batch_size=cfg.batch_size,
                 lr_critic=cfg.lr_critic,
                 lr_actor=cfg.lr_actor,
                 update_times=cfg.update_times,
                 input_type='vector',
                 input_feature=env.observation_space.shape[0],
                 device=cfg.device)

    # ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    score = 0.0
    training_steps = 0
    for i_episode in range(cfg.max_episodes):
        # change the probability of random action according to the training process
        epsilon = max(cfg.min_epsilon, cfg.max_epsilon - 0.01 * (i_episode / cfg.epsilon_decay))
        obs = env.reset()
        steps = 0
        epi_reward = 0.0
        done = False
        while not done:
            action = model.sampleAction(obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0

            # save the data into the replay buffer
            model.memory.insert((obs, action, reward / 100.0, next_obs, done_mask))

            # record the data
            score += reward
            training_steps += 1
            steps += 1
            epi_reward += reward

            obs = next_obs

        # learn the model according to the learning frequency.
        if i_episode >= cfg.learning_start and i_episode % cfg.learning_freq == 0:
            epi_critic_loss, epi_actor_loss = model.learn()
            epi_loss = epi_critic_loss + epi_actor_loss
            writer.add_scalar('critic_loss-episode', epi_critic_loss, global_step=i_episode)
            writer.add_scalar('critic_loss-training_steps', epi_critic_loss, global_step=training_steps)
            writer.add_scalar('actor_loss-episode', epi_actor_loss, global_step=i_episode)
            writer.add_scalar('actor_loss-training_steps', epi_actor_loss, global_step=training_steps)
            writer.add_scalar('loss-episode', epi_loss, global_step=i_episode)
            writer.add_scalar('loss-training_steps', epi_loss, global_step=training_steps)

        writer.add_scalar('steps-episode', steps, global_step=i_episode)
        writer.add_scalar('rewards-episode', epi_reward, global_step=i_episode)

        if i_episode % cfg.print_interval == 0 and i_episode > 0:
            print_and_log(
                "episode: {}, training steps: {}, avg score: {:.2f}, critic_loss: {:.5f}, actor_loss: {:.5f}, loss: {:.5f}, buffer size: {}, epsilon:{:.2f}%"
                .format(i_episode, training_steps, score / cfg.print_interval, epi_critic_loss,
                        epi_actor_loss, epi_loss, model.memory.size(), epsilon * 100), logger)
            score = 0.0

        # test the model with epsilon=0.0
        if i_episode % cfg.test_freq == 0 and i_episode > 0:
            test_score = 0.0
            for t in range(cfg.test_episodes):
                obs = env.reset()
                done = False
                while not done:
                    action = model.sampleAction(obs)
                    next_obs, reward, done, _ = env.step(action)

                    test_score += reward
                    obs = next_obs
            print_and_log('******************Test result******************', logger)
            print_and_log("avg score: {:.2f}".format(test_score / cfg.test_episodes), logger)
            print_and_log('***********************************************', logger)


def trainPPO(env, writer, logger):
    # create model
    model = PPO(action_space=env.action_space,
                clip_param=cfg.clip_param,
                value_loss_coef=cfg.value_loss_coef,
                entropy_coef=cfg.entropy_coef,
                learning_rate=cfg.learning_rate,
                update_times=cfg.update_times,
                input_type='vector',
                input_feature=env.observation_space.shape[0],
                max_grad_norm=cfg.max_grad_norm,
                device=cfg.device)

    score = 0.0
    training_steps = 0
    for i_episode in range(cfg.max_episodes):
        obs = env.reset()
        steps = 0
        epi_reward = 0.0
        done = False
        while not done:
            for t in range(cfg.T_horizon):
                with torch.no_grad():
                    action, prob_a = model.sampleAction(obs)
                next_obs, reward, done, _ = env.step(action)
                done_mask = 0.0 if done else 1.0

                # save the data into the replay buffer
                model.insert((obs, action, reward / 100.0, next_obs, prob_a.item(), done_mask))

                # record the data
                score += reward
                training_steps += 1
                steps += 1
                epi_reward += reward

                if done:
                    break
                obs = next_obs

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = model.learn()
            writer.add_scalar('value_loss-episode', value_loss_epoch, global_step=i_episode)
            writer.add_scalar('action_loss-episode', action_loss_epoch, global_step=i_episode)
            writer.add_scalar('dist_entropy-episode', dist_entropy_epoch, global_step=i_episode)

        writer.add_scalar('steps-episode', steps, global_step=i_episode)
        writer.add_scalar('rewards-episode', epi_reward, global_step=i_episode)

        if i_episode % cfg.print_interval == 0 and i_episode > 0:
            print_and_log(
                "episode: {}, training steps: {}, avg score: {:.2f}, value_loss: {:.5f}, action_loss: {:.5f}, dist_entropy: {:.5f}"
                .format(i_episode, training_steps, score / cfg.print_interval, value_loss_epoch,
                        action_loss_epoch, dist_entropy_epoch), logger)
            score = 0.0

        # test the model with epsilon=0.0
        if i_episode % cfg.test_freq == 0 and i_episode > 0:
            test_score = 0.0
            for t in range(cfg.test_episodes):
                obs = env.reset()
                done = False
                while not done:
                    action, _ = model.sampleAction(obs)
                    next_obs, reward, done, _ = env.step(action)

                    test_score += reward
                    obs = next_obs
            print_and_log('******************Test result******************', logger)
            print_and_log("avg score: {:.2f}".format(test_score / cfg.test_episodes), logger)
            print_and_log('***********************************************', logger)


if __name__ == '__main__':
    cfg.min_epsilon = 0.01
    cfg.max_epsilon = 0.5
    cfg.print_interval = 100
    cfg.learning_start = 100
    cfg.test_episodes = 100
    cfg.batch_size = 32
    cfg.learning_freq = 1

    # specific config
    if cfg.env == 'academy_empty_goal':
        cfg.test_freq = 2000
        cfg.epsilon_decay = 6000
    elif cfg.env == 'academy_3_vs_1_with_keeper':
        cfg.test_freq = 5000
        cfg.epsilon_decay = 30000

    # create output path
    cur_dir = os.path.abspath(os.curdir)
    root_output_path = os.path.join(cur_dir, 'output')
    if not os.path.exists(root_output_path):
        os.mkdir(root_output_path)
    # logger_path, final_output_path = create_logger(root_output_path, cfg)
    logger, final_output_path = create_logger(root_output_path, cfg)
    # logger = open(logger_path, 'w')
    writer = SummaryWriter(log_dir=os.path.join(final_output_path, 'tb'))

    print_and_log('******************Called with config******************', logger)
    print_and_log(cfg, logger)
    print_and_log('******************************************************', logger)

    # create env
    # env = gym.make('CartPole-v0')         # CartPole as a simple discrete task for testing the algorithm
    env = football_env.create_environment(env_name=cfg.env,
                                          representation='simple115',
                                          number_of_left_players_agent_controls=1,
                                          stacked=False,
                                          logdir='./output/',
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False,
                                          render=False)

    if cfg.model == 'DDPG':
        cfg.lr_critic = 0.005
        cfg.lr_actor = 0.0025
        trainDDPG(env, writer, logger)
    elif cfg.model == 'TD3':
        # trainTD3(env, writer, logger)
        pass
    elif cfg.model == 'PPO':
        cfg.value_loss_coef = 0.5
        cfg.entropy_coef = 0.01
        cfg.max_grad_norm = 0.5
        cfg.update_times = 4
        cfg.T_horizon = 32
        trainPPO(env, writer, logger)

    env.close()
    logger.close()
