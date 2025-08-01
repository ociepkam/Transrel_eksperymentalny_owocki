from psychopy import event, logging


def abort_with_error(err):
    logging.critical(err)
    raise Exception(err)


def check_exit(key='f7'):
    stop = event.getKeys(keyList=[key])
    if len(stop) > 0:
        logging.critical('Experiment finished by user! {} pressed.'.format(key))
        exit(1)
