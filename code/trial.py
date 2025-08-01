import os
from copy import deepcopy
from psychopy.visual import TextStim, ImageStim
import numpy as np
import random


def prepare_stim(win, trail_raw, config, stimulus_type):
    trial = deepcopy(trail_raw)
    if stimulus_type == "text":
        pos = [config["stimulus_pos"][0] - config["distance_between_stim_pairs"][0] / 2 - config["distance_in_pair"][0],
               config["stimulus_pos"][1] - config["distance_between_stim_pairs"][1] / 2 - config["distance_in_pair"][1]]
        for i, pair in enumerate(trial["stimulus"]):
            for c, elem in enumerate(pair):
                trial["stimulus"][i][c] = TextStim(win, color=config["text_color"], text=elem,
                                                   height=config["elements_size"], pos=pos)
                pos[0] += config["distance_in_pair"][0]
                pos[1] += config["distance_in_pair"][1]
            pos[0] += config["distance_between_stim_pairs"][0]
            pos[1] += config["distance_between_stim_pairs"][1]

        pos = [config["answers_pos"][0] - config["distance_between_answer_pairs"][0] * 1.5 - config["distance_in_pair"][0],
               config["answers_pos"][1] - config["distance_between_answer_pairs"][1] * 1.5 - config["distance_in_pair"][1]]
        for i, pair in enumerate(trial["pairs"]):
            for c, elem in enumerate(pair):
                trial["pairs"][i][c] = TextStim(win, color=config["text_color"], text=elem,
                                                height=config["elements_size"], pos=pos)
                pos[0] += config["distance_in_pair"][0]
                pos[1] += config["distance_in_pair"][1]
            pos[0] += config["distance_between_answer_pairs"][0]
            pos[1] += config["distance_between_answer_pairs"][1]
    elif stimulus_type == "image":
        pos = [config["stimulus_pos"][0] - config["distance_between_stim_pairs"][0] / 2 - config["distance_in_pair"][0],
               config["stimulus_pos"][1] - config["distance_between_stim_pairs"][1] / 2 - config["distance_in_pair"][1]]
        for i, pair in enumerate(trial["stimulus"]):
            for c, elem in enumerate(pair):
                if c == 0 or c == 2:
                    trial["stimulus"][i][c] = ImageStim(win=win, image=elem, size=config["images_size"], pos=pos)
                else:
                    trial["stimulus"][i][c] = TextStim(win, color=config["text_color"], text=elem, height=config["elements_size"], pos=pos)
                pos[0] += config["distance_in_pair"][0]
                pos[1] += config["distance_in_pair"][1]
            pos[0] += config["distance_between_stim_pairs"][0]
            pos[1] += config["distance_between_stim_pairs"][1]

        pos = [config["answers_pos"][0] - config["distance_between_answer_pairs"][0] * 1.5 - config["distance_in_pair"][0],
               config["answers_pos"][1] - config["distance_between_answer_pairs"][1] * 1.5 - config["distance_in_pair"][1]]
        for i, pair in enumerate(trial["pairs"]):
            for c, elem in enumerate(pair):
                if c == 0 or c == 2:
                    trial["pairs"][i][c] = ImageStim(win=win, image=elem, size=config["images_size"], pos=pos)
                else:
                    trial["pairs"][i][c] = TextStim(win, color=config["text_color"], text=elem, height=config["elements_size"], pos=pos)
                pos[0] += config["distance_in_pair"][0]
                pos[1] += config["distance_in_pair"][1]
            pos[0] += config["distance_between_answer_pairs"][0]
            pos[1] += config["distance_between_answer_pairs"][1]
        pass
    else:
        raise Exception(f"stimulus_type == {stimulus_type} is not implemented")
    return trial


def replace_stimulus_in_pair(pair, new_stimulus):
    new_pair = []
    for elem in pair:
        if elem in new_stimulus.keys():
            new_pair.append(new_stimulus[elem])
        else:
            new_pair.append(elem)
    return new_pair


def replace_stimulus(trial_raw, allowed_stimulus):
    trial = deepcopy(trial_raw)
    a, b, c = np.random.choice(allowed_stimulus, 3, replace=False)
    new_stimulus = {"A": a, "B": b, "C": c}
    trial["stimulus"] = [replace_stimulus_in_pair(pair, new_stimulus) for pair in trial["stimulus"]]
    trial["pairs"] = [replace_stimulus_in_pair(pair, new_stimulus) for pair in trial["pairs"]]
    trial["answer"] = replace_stimulus_in_pair(trial["answer"], new_stimulus)
    trial["order"] = [new_stimulus[elem] for elem in trial["order"]]
    return trial, [a, b, c]


def reverse_pair(pair):
    if pair[1] == "/":
        return f"{pair[2]}\\{pair[0]}"
    elif pair[1] == "\\":
        return f"{pair[2]}/{pair[0]}"
    else:
        return pair[::-1]


def all_possible_trials():
    all_trials = {"bind":    {"two_pairs": [], "sym_reversed": [], "sym_identical": [], "rel_reversed": [], "rel_identical": []},
                  "no_bind": {"two_pairs": [], "sym_reversed": [], "sym_identical": [], "rel_reversed": [], "rel_identical": []}}

    # binding = False
    # stimulus = [{"stim": ["A/B", r"C\B"], "order": "ABC"},
    #             {"stim": ["A/B", r"A\C"], "order": "CAB"},
    #             {"stim": [r"A\B", "C/B"], "order": "CBA"},
    #             {"stim": [r"A\B", "A/C"], "order": "BAC"}]
    # for i, elem in enumerate(stimulus):
    #     stim = elem["stim"]
    #     order = elem["order"]
    #     for answer_type in ["identical", "reversed", "two_pairs"]:
    #         for correct_pair in stim:
    #             if answer_type == "identical":
    #                 answer = correct_pair
    #             elif answer_type == "reversed":
    #                 answer = reverse_pair(correct_pair)
    #             elif answer_type == "two_pairs" and i == 0:
    #                 answer = f"{order[0]}/{order[2]}"
    #             else:
    #                 answer = f"{order[2]}\\{order[0]}"
    #             for incorrect_far in [f"{order[0]}\\{order[2]}", f"{order[2]}/{order[0]}"]:
    #                 for incorrect_pair in [f"{order[0]}\\{order[1]}", f"{order[1]}\\{order[2]}",
    #                                        f"{order[1]}/{order[0]}", f"{order[2]}/{order[1]}"]:
    #                     pairs = [answer, incorrect_pair, incorrect_far]
    #                     random.shuffle(pairs)
    #                     trial = {"stimulus": stim, "pairs": pairs, "answer": answer, "order": order,
    #                              "answer_type": answer_type, "with_binding": binding}
    #                     all_trials["bind"][answer_type].append(trial)

    binding = True
    stimulus = [{"stim": ["A/B",  "B|C"], "order": "ABC"},
                {"stim": ["A/B",  "A|C"], "order": "CAB"},
                {"stim": [r"A\B", "C|B"], "order": "CBA"},
                {"stim": [r"A\B", "C|A"], "order": "BAC"}]

    for i, elem in enumerate(stimulus):
        stim = elem["stim"]
        order = elem["order"]
        for answer_type in ["two_pairs", "rel_reversed", "rel_identical", "sym_reversed", "sym_identical"]:
            if answer_type == "sym_identical":
                answer = stim[1]
            elif answer_type == "rel_identical":
                answer = stim[0]
            elif answer_type == "sym_reversed":
                answer = reverse_pair(stim[1])
            elif answer_type == "rel_reversed":
                answer = reverse_pair(stim[0])
            elif answer_type == "two_pairs" and i == 0:
                answer = f"{order[0]}/{order[2]}"
            else:
                answer = f"{order[2]}\\{order[0]}"

            for incorrect_far in [f"{order[0]}\\{order[2]}", f"{order[2]}/{order[0]}"]:
                for incorrect_pair in [f"{stim[0][0]}|{stim[0][2]}", f"{stim[0][2]}|{stim[0][0]}"]:
                    pairs = [answer, incorrect_pair, incorrect_far]
                    trial = {"stimulus": stim, "pairs": pairs, "answer": answer, "order": order,
                             "answer_type": answer_type, "with_binding": binding}
                    all_trials["no_bind"][answer_type].append(trial)

            for incorrect_far in [f"{order[0]}|{order[2]}", f"{order[2]}|{order[0]}"]:
                for incorrect_pair in [f"{stim[0][0]}\\{stim[0][2]}", f"{stim[0][2]}/{stim[0][0]}",
                                       f"{stim[1][0]}\\{stim[1][2]}", f"{stim[1][0]}/{stim[1][2]}",
                                       f"{stim[1][2]}\\{stim[1][0]}", f"{stim[1][2]}/{stim[1][0]}"]:
                    pairs = [answer, incorrect_pair, incorrect_far]
                    random.shuffle(pairs)
                    trial = {"stimulus": stim, "pairs": pairs, "answer": answer, "order": order,
                             "answer_type": answer_type, "with_binding": binding}
                    all_trials["no_bind"][answer_type].append(trial)

    return all_trials

# For random generation
# class Trial:
#     def __init__(self, with_equal, memory=False, elements=("A", "B", "C"),
#                  answer_type=None, symbols=None, randomize_elements=True):
#         if symbols is None:
#             symbols = {"higher": "/", "lower": "\\", "equal": "|"}
#         if answer_type is None:
#             answer_type = random.choice(["two_pairs", "reversed", "identical"])
#         self.elements = elements
#         self.symbols = symbols
#         self.with_equal = with_equal
#         self.memory = memory
#         self.answer_type = answer_type
#         if randomize_elements:
#             random.shuffle(self.elements)
#
#         self.pairs = []
#         self.answers = []
#
#         if with_equal:
#             # stimulus
#             if random.random() < 0.5:
#                 self.pairs.append([self.elements[0], self.symbols["higher"], self.elements[1]])  # A/B
#                 if random.random() < 0.5:
#                     equal_pair = [self.elements[0], self.symbols["equal"], self.elements[2]]  # A|C
#                     self.order = [self.elements[2], self.elements[0], self.elements[1]]
#                 else:
#                     equal_pair = [self.elements[1], self.symbols["equal"], self.elements[2]]  # B|C
#                     self.order = [self.elements[0], self.elements[1], self.elements[2]]
#             else:
#                 self.pairs.append([self.elements[0], self.symbols["lower"], self.elements[1]])  # A\B
#                 if random.random() < 0.5:
#                     equal_pair = [self.elements[0], self.symbols["equal"], self.elements[2]]  # A|C
#                     self.order = [self.elements[1], self.elements[0], self.elements[2]]
#                 else:
#                     equal_pair = [self.elements[2], self.symbols["equal"], self.elements[1]]  # C|B
#                     self.order = [self.elements[2], self.elements[1], self.elements[0]]
#             self.pairs.append(equal_pair)
#             # answers
#             if self.answer_type == "identical":
#                 self.correct_answer = equal_pair
#             elif self.answer_type == "reversed":
#                 self.correct_answer = self.reverse_pair(equal_pair)
#             else:  # self.correct_answer == "two_pairs"
#                 self.correct_answer = random.choice([[self.order[0], self.symbols["higher"], self.order[2]],
#                                                      [self.order[2], self.symbols["lower"], self.order[0]]])
#             if random.random() < 0.5:
#                 incorrect_pair = random.choice([[self.pairs[0][0], self.symbols["equal"], self.pairs[0][2]],
#                                                 [self.pairs[0][2], self.symbols["equal"], self.pairs[0][0]]])
#                 incorrect_far = random.choice([[self.order[0], self.symbols["lower"], self.order[2]],
#                                                [self.order[2], self.symbols["higher"], self.order[0]]])
#             else:
#                 incorrect_pair = random.choice([self.pairs[0][::-1],
#                                                 self.reverse_pair(self.pairs[0])[::-1],
#                                                 [equal_pair[0], self.symbols["lower"], equal_pair[2]],
#                                                 [equal_pair[2], self.symbols["lower"], equal_pair[0]],
#                                                 [equal_pair[0], self.symbols["higher"], equal_pair[2]],
#                                                 [equal_pair[2], self.symbols["higher"], equal_pair[0]]])
#                 incorrect_far = random.choice([[self.order[0], self.symbols["equal"], self.order[2]],
#                                                [self.order[2], self.symbols["equal"], self.order[0]]])
#
#         else:
#             # stimulus
#             if random.random() < 0.5:
#                 self.pairs.append([self.elements[0], self.symbols["higher"], self.elements[1]])  # A/B
#                 if random.random() < 0.5:
#                     self.pairs.append([self.elements[2], self.symbols["lower"], self.elements[1]])  # C\B
#                     self.order = [self.elements[0], self.elements[1], self.elements[2]]
#                 else:
#                     self.pairs.append([self.elements[0], self.symbols["lower"], self.elements[2]])  # A\C
#                     self.order = [self.elements[2], self.elements[0], self.elements[1]]
#             else:
#                 self.pairs.append([self.elements[0], self.symbols["lower"], self.elements[1]])  # A\B
#                 if random.random() < 0.5:
#                     self.pairs.append([self.elements[2], self.symbols["higher"], self.elements[1]])  # C/B
#                     self.order = [self.elements[2], self.elements[1], self.elements[0]]
#                 else:
#                     self.pairs.append([self.elements[0], self.symbols["higher"], self.elements[2]])  # A/C
#                     self.order = [self.elements[1], self.elements[0], self.elements[2]]
#             # answers
#             if self.answer_type == "identical":
#                 self.correct_answer = random.choice(self.pairs)
#             elif self.answer_type == "reversed":
#                 self.correct_answer = self.reverse_pair(random.choice(self.pairs))
#             else:  # self.correct_answer == "two_pairs"
#                 self.correct_answer = random.choice([[self.order[0], self.symbols["higher"], self.order[2]],
#                                                      [self.order[2], self.symbols["lower"], self.order[0]]])
#
#             incorrect_pair = random.choice([[self.order[0], self.symbols["lower"], self.order[1]],
#                                             [self.order[1], self.symbols["lower"], self.order[2]],
#                                             [self.order[1], self.symbols["higher"], self.order[0]],
#                                             [self.order[2], self.symbols["higher"], self.order[1]]])
#
#             incorrect_far = random.choice([[self.order[0], self.symbols["lower"], self.order[2]],
#                                            [self.order[2], self.symbols["higher"], self.order[0]]])
#
#         self.answers = [incorrect_far, incorrect_pair, self.correct_answer]
#         random.shuffle(self.answers)
#
#     def reverse_pair(self, pair):
#         if pair[1] == self.symbols["lower"]:
#             return [pair[2], self.symbols["higher"], pair[0]]
#         elif pair[1] == self.symbols["higher"]:
#             return [pair[2], self.symbols["lower"], pair[0]]
#         else:
#             return [pair[2], self.symbols["equal"], pair[0]]
