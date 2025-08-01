import os


def choose_stimulus(stimulus_type):
    if stimulus_type == "Latin":
        import string
        alphabet = list(string.ascii_uppercase)
        to_remove = "IQMERW"
        [alphabet.remove(elem) for elem in to_remove]
        stimulus = {"type": "text", "strict_sets": False, "stimulus_list": alphabet}
    elif stimulus_type == "Numbers":
        stimulus = {"type": "text", "strict_sets": False, "stimulus_list": [str(num) for num in range(10)]}
    elif stimulus_type == "Figures":
        stimulus = {"type": "image", "strict_sets": True, "stimulus_list": [["square.png", "ellipse.png", "right_triangle.png"],
                                                                            ["rotated_square.png", "circle.png", "equilateral_triangle.png"],
                                                                            ["vertical_rectangle.png", "hexagon.png", "acute_triangle.png"],
                                                                            ["horizontal_rectangle.png", "star.png", "trapezoid.png"]]}
        for idx, image_set in enumerate(stimulus["stimulus_list"]):
            stimulus["stimulus_list"][idx] = [os.path.join('images', 'png', i) for i in image_set]
    elif stimulus_type == "Fruits":
        stimulus = {"type": "image", "strict_sets": False,
                    "stimulus_list": [os.path.join('images', 'fruits', image) for image in os.listdir(os.path.join('images', 'fruits'))]}
    else:
        raise Exception("Unknown stimulus type")
    return stimulus

# [["■", "⬬", "◣"],
#  ["◆", "⬤", "▲"],
#  ["▮", "⬣", "►"],
#  ["▬", "★", "▰"]]
