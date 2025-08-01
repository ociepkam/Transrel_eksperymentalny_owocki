import atexit
import time
from os.path import join
import random
import csv
from psychopy import visual, event, core

from code.show_info import part_info, show_info, show_image
from code.load_data import load_config
from code.screen_misc import get_screen_res
from code.trial import replace_stimulus, prepare_stim
from code.choose_stimulus import choose_stimulus
from code.check_exit import check_exit
from code.block import prepare_blocks

PART_ID = None
RESULTS = []
N = 0


@atexit.register
def save_beh_results():
    current_time = time.strftime("%Y_%m_%d_%H_%M", time.gmtime())
    with open(join('results', f'{PART_ID}_beh_{current_time}.csv'), 'w', newline='') as beh_file:
        dict_writer = csv.DictWriter(beh_file, RESULTS[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(RESULTS)


def draw_stim(stim, flag):
    for pair in stim:
        for elem in pair:
            elem.setAutoDraw(flag)


def run_block(block, config, win, stimulus_all, stimulus_type, fixation, clock, block_idx, block_type, break_text):
    global N, RESULTS
    stimulus_last = []
    for trial_raw in block:
        key = None
        reaction_time = None
        acc = -1
        N += 1

        # prepare trial stimulus
        if stimulus_all["strict_sets"]:
            stimulus_allowed = random.choice([elem for elem in stimulus_all["stimulus_list"] if sorted(elem) != sorted(stimulus_last)])
        else:
            stimulus_allowed = [elem for elem in stimulus_all["stimulus_list"] if elem not in stimulus_last]
        trial, stimulus_last = replace_stimulus(trial_raw, allowed_stimulus=stimulus_allowed)
        trial = prepare_stim(win, trial, config, stimulus_all["type"])

        # fixation
        fixation.setAutoDraw(True)
        win.flip()
        time.sleep(config["fixation_time"])
        fixation.setAutoDraw(False)
        win.flip()

        # draw trial
        draw_stim(trial["stimulus"], True)
        win.callOnFlip(clock.reset)
        win.flip()
        while clock.getTime() < config[f"{block_type}_thinking_time"]:
            check_exit()
            win.flip()
        draw_stim(trial["pairs"], True)
        win.callOnFlip(clock.reset)
        win.callOnFlip(event.clearEvents)
        win.flip()
        while clock.getTime() < config[f"{block_type}_answer_time"]:
            key = event.getKeys(keyList=config["reaction_keys"])
            if key:
                reaction_time = clock.getTime()
                key = key[0]
                break
            check_exit()
            win.flip()
        draw_stim(trial["stimulus"], False)
        draw_stim(trial["pairs"], False)
        win.callOnFlip(clock.reset)
        win.callOnFlip(event.clearEvents)
        win.flip()

        if key:
            acc = 1 if trial_raw["pairs"][config["reaction_keys"].index(key)] == trial_raw["answer"] else 0

        if block_type == "training":
            if acc == 1:
                feedback_text = u'Poprawna odpowied\u017A'
            elif acc == 0:
                feedback_text = u'Odpowied\u017A niepoprawna'
            else:
                feedback_text = u'Nie udzieli\u0142e\u015B odpowiedzi'

            feedback = visual.TextStim(win, text=feedback_text, color=config["feedback_color"], height=config["feedback_size"])
            feedback.setAutoDraw(True)
            win.flip()
            time.sleep(config["feedback_time"])
            feedback.setAutoDraw(False)
            win.flip()

        # results
        trial_results = {"n": N,
                         "block_type": block_type,
                         "block_n": block_idx,
                         "binding": trial_raw["with_binding"],
                         "trial_type": trial_raw["answer_type"],
                         "acc": acc,
                         "rt": reaction_time,
                         "order": trial_raw["order"],
                         "stimulus": trial_raw["stimulus"],
                         "answers": trial_raw["pairs"],
                         "correct_answer": trial_raw["answer"]}
        RESULTS.append(trial_results)

        # wait
        wait_time = config[f"{block_type}_wait_time"] + random.random() * config[f"{block_type}_wait_jitter"]
        while clock.getTime() < wait_time:
            check_exit()
            win.flip()

        # break
        break_text.setAutoDraw(True)
        win.flip()
        while True:
            key = event.getKeys(keyList=["space"])
            if key:
                break_text.setAutoDraw(False)
                break
            check_exit()

        win.flip()



def main():
    global PART_ID
    config = load_config()
    info, PART_ID = part_info()

    screen_res = dict(get_screen_res())
    win = visual.Window(list(screen_res.values()), fullscr=True, monitor='testMonitor', units='pix', screen=0,
                        color=config["screen_color"])
    event.Mouse(visible=False)
    clock = core.Clock()
    fixation = visual.TextStim(win, color=config["fixation_color"], text=config["fixation_text"],
                               height=config["fixation_size"])

    break_text = visual.TextStim(win, color=config["break_color"], text=config["break_text"],
                                 height=config["break_size"])

    training_blocks = prepare_blocks(config["training_n_blocks"], config["training_trials"])
    experiment_blocks = prepare_blocks(config["experiment_n_blocks"], config["experiment_trials"])
    stimulus_all = choose_stimulus(config["stimulus_type"])

    # --------------------- training ---------------------
    # show_info(win, join('.', 'messages', 'training_1.txt'), text_color=config["text_color"],
    #           text_size=config["text_size"], screen_res=screen_res)
    # show_info(win, join('.', 'messages', 'training_2.txt'), text_color=config["text_color"],
    #           text_size=config["text_size"], screen_res=screen_res)
    show_image(win, 'training_1.jpg', list(screen_res.values()))
    show_image(win, 'training_2.jpg', list(screen_res.values()))
    for block_idx, trials_list in enumerate(training_blocks):

        run_block(block=trials_list,
                  config=config,
                  win=win,
                  stimulus_all=stimulus_all,
                  stimulus_type=config["stimulus_type"],
                  fixation=fixation,
                  clock=clock,
                  block_idx=block_idx + 1,
                  block_type="training",
                  break_text=break_text)

        if block_idx < config["training_n_blocks"] - 1:
            show_info(win, join('.', 'messages', f'break.txt'), text_color=config["text_color"],
                      text_size=config["text_size"], screen_res=screen_res)

    # --------------------- experiment ---------------------
    # show_info(win, join('.', 'messages', f'experiment.txt'), text_color=config["text_color"],
    #           text_size=config["text_size"], screen_res=screen_res)
    show_image(win, 'experiment.jpg', list(screen_res.values()))

    for block_idx, trials_list in enumerate(experiment_blocks):

        run_block(block=trials_list,
                  config=config,
                  win=win,
                  stimulus_all=stimulus_all,
                  stimulus_type=config["stimulus_type"],
                  fixation=fixation,
                  clock=clock,
                  block_idx=block_idx + 1,
                  block_type="experiment",
                  break_text=break_text)

        if block_idx < config["experiment_n_blocks"] - 1:
            show_info(win, join('.', 'messages', f'break.txt'), text_color=config["text_color"],
                      text_size=config["text_size"], screen_res=screen_res)

    show_info(win, join('.', 'messages', 'end.txt'), text_color=config["text_color"],
              text_size=config["text_size"], screen_res=screen_res)


if __name__ == "__main__":
    main()
