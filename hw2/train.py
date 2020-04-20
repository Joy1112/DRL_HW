import os
import numpy as np
# import gym
import gfootball.env as football_env
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter

from dqn import DQN, DoubleDQN
from create_logger import create_logger, print_and_log, print_and_write


def train():
    cfg = edict()
    cfg.env = 'academy_empty_goal'      # 'academy_empty_goal' or 'academy_3_vs_1_with_keeper'
    cfg.model = 'DoubleDQN'             # 'DQN' or 'DoubleDQN' or 'DuelingDQN'

    # common config
    cfg.device = 'cuda:0'
    cfg.min_epsilon = 0.01
    cfg.max_epsilon = 0.10
    cfg.print_interval = 100
    cfg.learning_start = 100
    cfg.test_episodes = 100
    cfg.gamma = 0.98
    cfg.batch_size = 32
    cfg.update_times = 5
    cfg.target_q_update_freq = 50

    # specific config
    if cfg.env == 'academy_empty_goal':
        cfg.max_episodes = 100000
        cfg.learning_freq = 1
        cfg.test_freq = 2000
        cfg.learning_rate = 0.0003
        cfg.buffer_size = 100000
        cfg.epsilon_decay = 6000
    elif cfg.env == 'academy_3_vs_1_with_keeper':
        cfg.max_episodes = 500000
        cfg.learning_freq = 5
        cfg.test_freq = 5000
        cfg.learning_rate = 0.0005
        cfg.buffer_size = 200000
        cfg.epsilon_decay = 30000

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
    env = football_env.create_environment(env_name=cfg.env,
                                          representation='simple115',
                                          number_of_left_players_agent_controls=1,
                                          stacked=False,
                                          logdir='./output/',
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False,
                                          render=False)

    # env = gym.make('CartPole-v1')

    if cfg.model == 'DQN':
        model = DQN(num_actions=19,
                    gamma=cfg.gamma,
                    buffer_size=cfg.buffer_size,
                    batch_size=cfg.batch_size,
                    learning_rate=cfg.learning_rate,
                    update_times=cfg.update_times,
                    target_q_update_freq=cfg.target_q_update_freq,
                    input_type='vector',
                    input_feature=115,
                    random_action=True,
                    device=cfg.device)
    elif cfg.model == 'DoubleDQN':
        model = DoubleDQN(num_actions=2,
                          gamma=cfg.gamma,
                          buffer_size=cfg.buffer_size,
                          batch_size=cfg.batch_size,
                          learning_rate=cfg.learning_rate,
                          update_times=cfg.update_times,
                          target_q_update_freq=cfg.target_q_update_freq,
                          input_type='vector',
                          input_feature=4,
                          random_action=True,
                          device=cfg.device)
    elif cfg.model == 'DuelingDQN':
        pass

    score = 0.0
    training_steps = 0
    for i_episode in range(cfg.max_episodes):
        epsilon = max(cfg.min_epsilon, cfg.max_epsilon - 0.01 * (i_episode / cfg.epsilon_decay))
        obs = env.reset()
        steps = 0
        epi_reward = 0.0
        done = False
        while not done:
            action = model.sampleAction(obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            model.memory.insert((obs, action, reward, next_obs, done_mask))

            score += reward
            training_steps += 1
            steps += 1
            epi_reward += reward
            obs = next_obs

        if i_episode >= cfg.learning_start and i_episode % cfg.learning_freq == 0:
            epi_loss = model.learn()
            writer.add_scalar('loss-episode', epi_loss, global_step=i_episode)
            writer.add_scalar('loss-training_steps', epi_loss, global_step=training_steps)

        writer.add_scalar('steps-episode', steps, global_step=i_episode)
        writer.add_scalar('rewards-episode', epi_reward, global_step=i_episode)

        if i_episode % cfg.print_interval == 0 and i_episode > 0:
            print_and_write(
                "episode: {}, training steps: {}, avg score: {:.2f}, loss: {:.5f}, buffer size: {}, epsilon:{:.2f}%"
                .format(i_episode, training_steps, score / cfg.print_interval,
                        epi_loss, model.memory.size(), epsilon * 100), logger)
            score = 0.0

        if i_episode % cfg.test_freq == 0 and i_episode > 0:
            test_score = 0.0
            for t in range(cfg.test_episodes):
                epsilon = 0.0
                obs = env.reset()
                done = False
                while not done:
                    action = model.sampleAction(obs, epsilon)
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
