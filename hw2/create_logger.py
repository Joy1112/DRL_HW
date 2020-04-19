import os
import logging
import time


def create_logger(root_output_path, config):
    # set up logger
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    cfg_name = 'lr_' + str(config.learning_rate) + \
               '_action_' + str(mdp_action) + '_limitH_' + str(limit_H) + '_rhoStar_' + str(rho_star)
    final_output_path = os.path.join(root_output_path, cfg_name)
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    final_output_path = os.path.join(final_output_path, time.strftime('%Y-%m-%d-%H-%M'))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, final_output_path


def print_and_log(string, logger):
    print(string)
    if logger:
        logger.info(string)
