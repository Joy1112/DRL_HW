import os
import numpy as np
import random
import gym
# import gfootball.env as football_env
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter

from ddpg import OrnsteinUhlenbeckNoise, DDPG
from create_logger import create_logger, print_and_log, print_and_write


parser = OptionParser()
parser.add_option("-m", "--model", dest="model", type="str", help="The model/algorithm which to be used. (DDPG, TD3 or PPO)", default='DDPG')
parser.add_option("-e", "--env", dest="env", type="str", help="Which environment to run the experiment. (academy_3_vs_1_with_keeper or academy_empty_goal)", default='academy_3_vs_1_with_keeper')
parser.add_option("-g", "--gamma", dest="gamma", type="float", help="The discount factor", default=0.99)
parser.add_option("-t", "--tau", dest="tau", type="float", help="The factor for target network soft update", default=0.005)
parser.add_option("-b", "--buffer_size", dest="buffer_size", type="int", help="The size of the replay buffer", default=200000)
parser.add_option("--device", dest="device", type="str", help="The device to run the experiment. (cuda or cpu)", default='cpu')
parser.add_option("--update_times", dest="update_times", type="int", help="The number of updates every episode", default=10)
parser.add_option("--max_episodes", dest="max_episodes", type="int", help="The maximum number of episodes", default=500000)

cfg, in_args = parser.parse_args()


def train():
    cfg.min_epsilon = 0.01
    cfg.max_epsilon = 0.5
    cfg.print_interval = 100
    cfg.learning_start = 100
    cfg.test_episodes = 100
    cfg.batch_size = 32
    cfg.learning_freq = 1
    cfg.lr_critic = 0.005
    cfg.lr_actor = 0.0025

    # specific config
    if cfg.env == 'academy_empty_goal':
        cfg.test_freq = 2000
        cfg.epsilon_decay = 6000
    elif cfg.env == 'academy_3_vs_1_with_keeper':
        cfg.test_freq = 5000
        cfg.epsilon_decay = 6000

    # create output path
    cur_dir = os.path.abspath(os.curdir)
    root_output_path = os.path.join(cur_dir, 'output')
    if not os.path.exists(root_output_path):
        os.mkdir(root_output_path)
    logger_path, final_output_path = create_logger(root_output_path, cfg)
    logger = open(logger_path, 'w')
    writer = SummaryWriter(log_dir=os.path.join(final_output_path, 'tb'))

    print('******************Called with config******************')
    print(cfg)
    print('******************************************************')

    # create env
    env = gym.make('CartPole-v0')
    # env = football_env.create_environment(env_name=cfg.env,
    #                                       representation='simple115',
    #                                       number_of_left_players_agent_controls=1,
    #                                       stacked=False,
    #                                       logdir='./output/',
    #                                       write_goal_dumps=False,
    #                                       write_full_episode_dumps=False,
    #                                       render=False)

    # create model
    if cfg.model == 'DDPG':
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
    elif cfg.model == 'TD3':
        pass
    elif cfg.model == 'PPO':
        pass
    
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
            print_and_write(
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
            print_and_write('******************Test result******************', logger)
            print_and_write("avg score: {:.2f}".format(test_score / cfg.test_episodes), logger)
            print_and_write('***********************************************', logger)
    env.close()
    logger.close()


if __name__ == '__main__':
    train()
