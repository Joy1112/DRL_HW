import os
import logging
import time


def create_logger(root_output_path, config):
    """
    create the logger path and the data output path.
    """
    # set up logger
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    model_name = 'model_' + config.model
    final_output_path = os.path.join(root_output_path, model_name)
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    env_name = 'env_' + config.env
    final_output_path = os.path.join(final_output_path, env_name)
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    final_output_path = os.path.join(final_output_path, time.strftime('%Y-%m-%d-%H-%M'))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    file_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))

    # head = '%(asctime)-15s %(message)s'
    head = '%(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, file_name), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    return logger, final_output_path


def print_and_log(string, logger):
    print(string)
    if logger:
        logger.warning(string)


def print_and_write(string, file=None):
    """
    Here use the write function instead of the logging package because that google_football also uses the logging,
    and it will record every step information in the logger file.
    """
    print(string)
    if file:
        file.writelines(string)
        file.writelines('\n')
