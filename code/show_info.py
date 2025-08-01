from code.load_data import read_text_from_file
from psychopy import visual, gui, event
from code.check_exit import abort_with_error
import os


def part_info():
    info = {'Part_id': '', 'Part_age': '20', 'Part_sex': ['MALE', "FEMALE"]}
    dictDlg = gui.DlgFromDict(dictionary=info, title='Transrel experimental')
    if not dictDlg.OK:
        exit(1)
    return info, f"{info['Part_id']}_{info['Part_sex']}_{info['Part_age']}"


def show_info(win, file_name, text_size, text_color, screen_res, insert=''):
    msg = read_text_from_file(file_name, insert=insert)
    msg = visual.TextStim(win, color=text_color, text=msg, height=text_size,
                          wrapWidth=screen_res['width'], font='Segoe UI Emoji')
    msg.draw()
    win.flip()
    key = event.waitKeys(keyList=['f7', 'return', 'space'])
    if key == ['f7']:
        abort_with_error('Experiment finished by user on info screen! F7 pressed.')
    win.flip()

def show_image(win, file_name, size, key='f7'):
    image = visual.ImageStim(win=win, image=os.path.join('messages', file_name), interpolate=True, size=size)
    image.draw()
    win.flip()
    clicked = event.waitKeys(keyList=[key, 'return', 'space'])
    if clicked == [key]:
        exit(0)
    win.flip()