import os
import numpy as np
import gfootball.env as football_env
from torch.utils.tensorboard import SummaryWriter

from dqn import DQN
from create_logger import create_logger, print_and_log


def trainDQN():
    cfg = {}
    cfg['device'] = 'cpu'
    cfg['min_epsilon'] = 0.01
    cfg['max_epsilon'] = 0.08
    cfg['max_episodes'] = 10000
    cfg['print_interval'] = 20
    cfg['learning_start'] = 5
    cfg['learning_freq'] = 1
    cfg['test_freq'] = 100
    cfg['test_episodes'] = 10

    cur_dir = os.path.abspath(os.curdir)
    root_output_path = os.path.join(cur_dir, 'output')
    if not os.path.exists(root_output_path):
        os.mkdir(root_output_path)
    final_output_path = create_logger(root_output_path, )
    writer = SummaryWriter(log_dir=os.path.join(final_output_path, 'tb'))

    env = football_env.create_environment(env_name="academy_empty_goal",
                                          representation='simple115',
                                          number_of_left_players_agent_controls=1,
                                          stacked=False,
                                          logdir='./output/',
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False,
                                          render=False)

    model = DQN(num_actions=19,
                gamma=0.98,
                buffer_size=100000,
                batch_size=32,
                learning_rate=0.0005,
                update_times=5,
                target_q_update_freq=50,
                input_type='vector',
                input_feature=115,
                random_action=True,
                device=cfg.device)

    score = 0.0
    training_steps = 0
    for i_episode in range(cfg.max_episodes):
        epsilon = max(cfg.min_epsilon, cfg.max_epsilon - 0.01 * (i_episode / 200))
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
            print("episode: {}, avg score: {:.2f}, loss: {:.2f}, buffer size: {}, epsilon:{:.2f}%".format(
                i_episode, score / cfg.print_interval, epi_loss, model.memory.size(), epsilon * 100))
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
            print('******************Test result******************')
            print("avg score: {:.2f}".format(test_score / cfg.test_episodes))
            print('***********************************************')
    env.close()
