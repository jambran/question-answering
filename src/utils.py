import logging
import os

import mxnet as mx

__all__ = ['logging_config',
           'log_experiment_parameters',
           'format_epoch_updates',
           'check_for_gpu',
           ]


def logging_config(folder=None, name=None,
                   level=logging.INFO,
                   console_level=logging.INFO,
                   no_console=False):
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + '.log')
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def log_experiment_parameters(ARGS, logger):
    """
    to be used at the beginning of each experiment to capture exactly what hyperparameters the experiment used
    :param ARGS: parsed arguments
    :param logger: the logger that will log each arguemnt in ARGS at info level
    :return: None
    """
    for argument, value in vars(ARGS).items():
        logger.info(f"\t{argument:>20} = {value}")
    return


def format_epoch_updates(epoch_formatting_space=12,
                         train_loss_space=15.4,
                         train_acc_space=10.4,
                         val_acc_space=10.4):
    """
    get formatting strings to show the Epoch number, Train loss, and Val Accuracy
    :param epoch_formatting_space: desired formatting for epoch loss
    :param train_loss_space: desired formatting for train loss
    :param train_acc_space: desired formatting for accuracy on train set
    :param val_acc_space: desired formatting for accuracy on validation set
    :return:
    """
    formatting_log = f"%{epoch_formatting_space}d " \
        f"%{train_loss_space}f " \
        f"%{train_acc_space}f " \
        f"%{val_acc_space}f"

    formatting_header = f"%{epoch_formatting_space}s " \
        f"%{int(train_loss_space)}s " \
        f"%{int(train_acc_space)}s " \
        f"%{int(val_acc_space)}s"
    return formatting_header, formatting_log


def check_for_gpu(logger=None):
    if mx.test_utils.list_gpus():
        if logger is not None: logger.info("Using GPU")
        ctx = mx.gpu()
    else:
        if logger is not None: logger.info("Using CPU")
        ctx = mx.cpu()
    return ctx
