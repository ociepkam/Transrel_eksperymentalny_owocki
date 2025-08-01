from code.trial import all_possible_trials
import random
import numpy as np


def prepare_blocks(n_blocks, trials_types):
    all_trials = all_possible_trials()
    trials = []
    for bind_key, bind_value in all_trials.items():
        for type_key, type_value in bind_value.items():
            n_to_choose = trials_types[bind_key][type_key]
            chosen_trials = random.sample(type_value, n_to_choose)
            trials += chosen_trials
    random.shuffle(trials)
    return np.array_split(np.array(trials), n_blocks)
