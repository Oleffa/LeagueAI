import numpy as np
import cv2

def find_player_character(object_id, objects, confidence):
    cur_smallest = -1
    for i,o in enumerate(objects):
        if o.object_class == object_id and o.confidence > confidence:
            cur_smallest = i

    if cur_smallest >= 0:
        return objects[cur_smallest]
    else:
        return None

def object_exists(object_id, object_list, confidence):
    raise NotImplementedError


def get_player_hp(frame):
    raise NotImplementedError

def draw_bb(frame, x, y, x_min, y_min, x_max, y_max):
    cv2.circle(frame, (x, y), 5, (255,0,0), -1)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)
    return frame

class Player():
    def __init__(self, xPos, yPos, object_class):
        self.x = int(xPos)
        self.y = int(yPos)
        self.object_class = int(object_class)
    def update(self, xPos, yPos, object_class):
        self.x = int(xPos)
        self.y = int(yPos)
        self.object_class = int(object_class)


