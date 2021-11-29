import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import path
import yaml


def mark_point(image, x, y, color=(255, 255, 0), size=2, show=True, cross=True):
    """
    marks a point on the image at the (x,y) position and displays it
    """
    ran = list(range(-size, size + 1))
    # for i in ran:
    #     for j in ran:
    #         if not cross or i == j or i == -j:
    #             if 0 < x + i < len(image) and \
    #                 0 < y + j < len(image[0]):
    #                 image[x + i, y + j] = color
    if show:
        plt.imshow(image)
        plt.show()


def draw_rectangles(image, rectangles, color=(255, 0, 0), show=True):
    # image = image.copy()
    for rect in rectangles:
        x, y, w, h = rect
        # image[y, x] = color
        # image[y+h, x+w] = color
        for i in range(-1, w + 1):
            if x + i < len(image) and y + h < len(image[0]):
                image[x + i, y - 1] = color
                image[x + i, y + h] = color
        for j in range(-1, h + 1):
            if y + j < len(image[0]):
                image[x - 1, y + j] = color
                image[x + w, y + j] = color
    if show:
        plt.imshow(image)
        plt.show()


_reorder = lambda rect: (rect[1], rect[0], rect[3], rect[2])


def find_objects(image, color, size=(15, 15), tol_s=15, position=None, tol_p=2,
                 min_distance=2, splitted_objects=False,
                 mark_objects=False):
    """
    image: image to detects objects from
    color: fixed color of the object
    size: presuposed size
    tol_s: tolerance on the size
    position: presuposed position
    tol_p: tolerance on the position
    min_distance: minimal distance between two detected objects
    """
    # import ipdb; ipdb.set_trace()
    mask = cv2.inRange(image, np.array(color), np.array(color))
    output = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 1)
    detected = []
    rects = [_reorder(cv2.boundingRect(cnt)) for cnt in contours]
    if splitted_objects:
        rects = merge_rects(rects, 4)
    for rec in rects:
        x, y, w, h = rec
        if size is not None:
            if not assert_in((h, w), size, tol_s):
                continue
        if position is not None:
            if not assert_in((x, y), position, tol_p):
                continue
        if min_distance is not None:
            too_close = False
            for det in detected:
                if abs(det[0][0] - y) + abs(det[0][1] - x) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
        if mark_objects:
            draw_rectangles(image, [(x, y, w, h)], show=False)
        detected.append(((x, y), image[x:x+w, y:y+h]))
    return detected


def assert_in(observed, target, tol):
    if isinstance(target, int):
        target = (target, target)
    if isinstance(tol, int):
        tol = (tol, tol)
    return np.all([target[i] + tol[i] > observed[i] > target[i] - tol[i] for i in range(2)])


def repeat_upsample(rgb_array, k=4, l=4, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def _are_close(base, neib, thd):
    x0, y0, w0, h0 = base
    x1, y1, w1, h1 = neib
    if x0 <= x1 <= x0 + w0:
        if abs(y0 + h0 - y1) < thd or abs(y1 + h1 - y0) < thd:
            return True
    if y0 <= y1 <= y0 + h0:
        if abs(x0 + w0 - x1) < thd or abs(x1 + w1 - x0) < thd:
            return True
    if x0 <= x1 <= x0 + w0 and y0 <= y1 <= y0 + h0:
        return True
    return False


def merge_rects(rects, threshold=4):   #  x, y, w, h
    rects.sort()
    acceptedRects = []
    r = 0
    rem_rects = len(rects)
    while r < rem_rects:
        base = rects[r]
        x0, y0, w0, h0 = base
        n = 1
        while n < len(rects[r:]):
            neib = rects[r + n]
            x1, y1, w1, h1 = neib
            if _are_close(base, neib, threshold) or _are_close(neib, base, threshold):
                base = (x0, y0, max(x1 + w1 - x0, x0 + w0 - x1), max(y1 + h1 - y0, y0 + h0 - y1))
                rects.pop(r+n)
                rem_rects -= 1
            else:
                n += 1
        acceptedRects.append(base)
        r += 1
    return acceptedRects


def load_game_dict(game):
    def _str_to_tuple(loaded_dict):
        for key, value in loaded_dict.items():
            if isinstance(value, str) and value[0] == "(":
                loaded_dict[key] = eval(value)
            elif isinstance(value, dict):
                _str_to_tuple(value)
    config_path = path.join(path.dirname(__file__),
                            "games_configs.yaml")
    with open(config_path, "r") as config_file:
        game_dict = yaml.safe_load(config_file)[game.lower()]
    _str_to_tuple(game_dict)  # convert strings of tuples to tuples
    if "obj_size" not in game_dict:
        game_dict["obj_size"] = None
    if "splitted_objects" not in game_dict:
        game_dict["splitted_objects"] = False
    return game_dict
