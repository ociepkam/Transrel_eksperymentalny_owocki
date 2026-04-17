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


def _pair_parts(pair):
    if isinstance(pair, str):
        return pair[0], pair[1], pair[2]
    return pair[0], pair[1], pair[2]


def _invert_symbol(sym):
    if sym == "/":
        return "\\"
    if sym == "\\":
        return "/"
    return "|"


def canonical_pair(pair):
    left, sym, right = _pair_parts(pair)
    if left <= right:
        return (left, sym, right)
    return (right, _invert_symbol(sym), left)


def infer_all_relations(stim):
    """
    Given a stimulus (list of 2 relation strings, e.g. ["A/B", "B|C"]),
    infer all logically true relations among the three elements and return
    them as a set of canonical pairs.

    Works by assigning a numeric height to each element based on the
    relations in the stimulus (higher value = lower in hierarchy, following
    the convention that '\\' means "is lower than" and '/' means "is higher than").
    The first pair anchors two elements; the second pair places the third
    element relative to one of the already-anchored elements.
    All pairwise relations are then derived from the resulting height values
    and stored in canonical form to allow direction-independent comparison.

    Example:
        stim = ["A/B", "B|C"]
        → heights: A=0, B=1, C=1
        → true relations: {canonical(A/B), canonical(A/C), canonical(B|C)}
    """
    heights = {}
    l0, s0, r0 = _pair_parts(stim[0])
    heights[l0] = 0
    heights[r0] = heights[l0] + (1 if s0 == '\\' else -1 if s0 == '/' else 0)
    l1, s1, r1 = _pair_parts(stim[1])
    anchor, new = (l1, r1) if l1 in heights else (r1, l1)
    sym_dir = s1 if anchor == l1 else _invert_symbol(s1)
    heights[new] = heights[anchor] + (1 if sym_dir == '\\' else -1 if sym_dir == '/' else 0)
    true_rels = set()
    elems = list(heights.keys())
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            a, b = elems[i], elems[j]
            ha, hb = heights[a], heights[b]
            sym = '/' if ha > hb else '\\' if ha < hb else '|'
            true_rels.add(canonical_pair(f"{a}{sym}{b}"))
    return true_rels


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

            true_rels = infer_all_relations(stim)

            for incorrect_far in [f"{order[0]}\\{order[2]}", f"{order[2]}/{order[0]}"]:
                for incorrect_pair in [f"{stim[0][0]}|{stim[0][2]}", f"{stim[0][2]}|{stim[0][0]}"]:
                    pairs = [answer, incorrect_pair, incorrect_far]

                    can = [canonical_pair(p) for p in pairs]
                    if len(set(can)) < 3:
                        continue
                    if any(canonical_pair(p) in true_rels for p in [incorrect_pair, incorrect_far]):
                        continue

                    random.shuffle(pairs)
                    trial = {"stimulus": stim, "pairs": pairs, "answer": answer, "order": order,
                             "answer_type": answer_type, "with_binding": binding}
                    all_trials["no_bind"][answer_type].append(trial)

            for incorrect_far in [f"{order[0]}|{order[2]}", f"{order[2]}|{order[0]}"]:
                for incorrect_pair in [f"{stim[0][0]}\\{stim[0][2]}", f"{stim[0][2]}/{stim[0][0]}",
                                       f"{stim[1][0]}\\{stim[1][2]}", f"{stim[1][0]}/{stim[1][2]}",
                                       f"{stim[1][2]}\\{stim[1][0]}", f"{stim[1][2]}/{stim[1][0]}"]:
                    pairs = [answer, incorrect_pair, incorrect_far]
                    can = [canonical_pair(p) for p in pairs]
                    if len(set(can)) < 3:
                        continue
                    if any(canonical_pair(p) in true_rels for p in [incorrect_pair, incorrect_far]):
                        continue

                    random.shuffle(pairs)
                    trial = {"stimulus": stim, "pairs": pairs, "answer": answer, "order": order,
                             "answer_type": answer_type, "with_binding": binding}
                    all_trials["no_bind"][answer_type].append(trial)
    print(all_trials)
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
