import numpy as np
import cv2
import math
import random
import win32api, win32con


from PIL import ImageFont, ImageDraw, Image

class Policy:
    def __init__(self):
        self.theta = np.zeros((4,1))
        self.theta[0] = 1 # Theta for tower attack probability
        self.theta[1] = 1 # Theta for minion attack probability
        self.theta[2] = 1 # Theta for retreat probability
        self.theta[3] = 1


class Player():
    def __init__(self, xPos, yPos, object_class):
        self.x = int(xPos)
        self.y = int(yPos)
        self.w = 0
        self.h = 0
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.detected = False
        self.confidence = 0.0
        self.object_class = int(object_class)
        self.hp = 100.0
        self.policy = Policy()
        self.actions = ['Attack Tower', 'Attack Canon', 'Attack Caster', 'Attack Melee', 'Pushing', 'Retreating']
        font_mono = "./fonts/Syne-Mono.ttf"
        font_regular = "./fonts/Syne-Regular.ttf"
        self.font40 = ImageFont.truetype(font_regular, 40)
        self.font30 = ImageFont.truetype(font_regular, 34)

    def update(self, xPos, yPos, width, height, confidence, object_class):
        self.x = int(xPos)
        self.y = int(yPos)
        self.w = int(width)
        self.h = int(height)
        self.x_min = int(self.x - 0.5 * self.w)
        self.y_min = int(self.y - 0.5 * self.h)
        self.x_max = int(self.x + 0.5 * self.w)
        self.y_max = int(self.y + 0.5 * self.h)
        self.confidence = confidence
        self.object_class = int(object_class)

    def click_xy(self, x, y, button):
        SCREEN_WIDTH = 1920
        SCREEN_HEIGHT=1080
        TOP_OFFSET = 20
        win32api.SetCursorPos((x,y+TOP_OFFSET))
        if button == 0:
            print("Attack Move!")
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        elif button == 1:
            print("Move!")
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)
        elif button == 2:
            print("Wheel!")
        else:
            print("Unknown Button Click!")
        return

    def draw_bb(self, frame, color):
        cv2.circle(frame, (self.x, self.y), 5, (255,0,0), -1)
        cv2.rectangle(frame, (self.x_min, self.y_min), (self.x_max, self.y_max), color, 2)
        return frame

    def compute_hp(self, frame, draw_hp_box):
        width = 415
        x_offset = 71
        height = 15
        bottom_offset = 50
        x_center = np.shape(frame)[1]/2 - x_offset
        y_center = np.shape(frame)[0] - bottom_offset
        hp_x_min = int(x_center-width/2)
        hp_x_max = int(x_center+width/2)
        hp_y_min = int(y_center)
        hp_y_max = int(y_center+height)

        green_pixels = 0
        all_pixels = ((hp_x_max-hp_x_min) * (hp_y_max-hp_y_min))
        for x_hp in range(hp_x_min, hp_x_max, 1):
            for y_hp in range(hp_y_min, hp_y_max, 1):
                # Make sure we do not count any black pixels
                if frame[y_hp][x_hp][1] > 10:
                    green_pixels += 1
        hp = round(green_pixels/all_pixels, 2) * 100
        self.hp = hp
        if draw_hp_box:
            cv2.rectangle(frame, (hp_x_min, hp_y_min), (hp_x_max, hp_y_max), (0, 0, 255), 2)
            label = "HP: {} %".format(self.hp)
            font_size = 2
            font_thickness = 2
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)
            cv2.putText(frame, label, (int(hp_x_min +10), int(y_center + t_size[0][1])), cv2.FONT_HERSHEY_PLAIN,
                        font_size, [0, 0 , 255], font_thickness)
        return hp, frame

    def find_player_character(self, object_id, objects, confidence):
        cur_smallest = -1
        for i,o in enumerate(objects):
            if o.object_class == object_id and o.confidence > confidence:
                cur_smallest = i

        if cur_smallest >= 0:
            # Found character is a detection object as defined in LeagueAI_helper
            found_character = objects[cur_smallest]
            self.update(found_character.x, found_character.y, found_character.w, found_character.h, found_character.confidence,
                        found_character.object_class)
            self.detected = True
            return True
        else:
            self.detected = False
            return False

    def get_distance(self, o):
        return math.sqrt(math.pow(o.x - self.x, 2) + math.pow(o.y - self.y,2))

    def get_shortest(self, l):
        """
        This function gets a list of objects and returns the closest object.
        Useful to find the closest entity of a given object class
        :param l:
        :return:
        """
        if len(l) == 1:
            o = l[0]
            return l[0], self.get_distance(o)
        cur_min = 1000000
        cur = None
        for o in l:
            d = self.get_distance(o)
            if d < cur_min:
                cur_min = d
                cur = o
        return cur, cur_min

    def attack_prob(self, object_class, shortest_distance, hp):
        print(shortest_distance)
        TODO check the functions here
        """
        if shortest_distance > 0:
            shortest_distance = shortest_distance * 0.0075
            if object_class == 0:
                p = self.policy.theta[0][0] * hp * (1 / (1 + math.exp(-2 * (shortest_distance + 2))) -
                                            1 / (1 + math.exp(-2 * (shortest_distance - 4.6))))
                p /= 94.78
                if p > 1:
                    return 1
                elif p < 0:
                    return 0
                else:
                    return p
            if object_class == 1 or object_class == 2 or object_class == 3:
                p = self.policy.theta[1][0] * hp * (1 / (1 + math.exp(-2 * (shortest_distance - 3))) -
                                            1 / (1 + math.exp(-3 * (shortest_distance - 6))))
                p /= 94.78
                if p > 1:
                    return 1
                elif p < 0:
                    return 0
                else:
                    return p
        else:
            return 0
        """

    def retreat_prob(self, closest_enemy, hp):
        if closest_enemy > 0:
            x = hp * self.policy.theta[2][0]
            x2 = closest_enemy * self.policy.theta[2][0]
            p = 1-(0.5 * math.log((x2+1), 5000)) - (0.5 * (x/100.0))
            if p > 1:
                return 1
            elif p < 0:
                return 0
            else:
                return p
        else:
            return 0

    def decide_action(self, tower_prob, canon_prob, caster_prob, melee_prob, push_prob, retreat_prob):
        probs = [tower_prob, canon_prob, caster_prob, melee_prob, push_prob, retreat_prob]
        actions = np.arange(0, len(probs), 1)
        cutoffs = []
        for i in range(0, len(probs)):
            if sum(probs) <= 0:
                cutoffs.append(0)
            else:
                cutoffs.append(sum(probs[:i+1])/sum(probs))

        rnd = random.random()
        # -1 do nothing
        action = -1
        for i in range(0, len(cutoffs)):
            if rnd < cutoffs[i]:
                action = actions[i]
                break
        return action

    def show_probs(self, frame, tower_prob, canon_prob, caster_prob, melee_prob, push_prob, retreat_prob, action):
        colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0), (255, 0 ,0), (125, 125, 125)]
        input_probs = [tower_prob, canon_prob, caster_prob, melee_prob, push_prob, retreat_prob]
        probs = [0] * len(input_probs)
        if sum(input_probs) > 0:
            probs = []
            for i in range(0, len(input_probs)):
               probs.append(float(input_probs[i])/sum(input_probs))
        size = (300, 500)
        padding_right = 10
        # Draw rect
        frame = cv2.rectangle(frame, (0, 0), size, (48, 53, 57), -1)
        frame = cv2.rectangle(frame, (0, 0), (299, 499), (30, 30, 30), 2)
        bar_max_size = 280
        bar_height = 40
        for i in range(0, len(probs)):
            frame = cv2.rectangle(frame, (padding_right, 100 + int((bar_height+10) * i)),
                                  (padding_right + int(bar_max_size*probs[i]),
                                   100 + int((bar_height+10) * i) + bar_height),
                                  colors[i], -1)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((padding_right, 40), "Action probs.", font=self.font40, fill=(255,255,255,255))
        # Show probability bars
        for i in range(0, len(probs)):
            draw.text((padding_right, 100 + int((bar_height+10) * i)), self.actions[i], font=self.font30, fill=(255,255,255,255))
        draw.text((padding_right, 100 + int((bar_height+10) * (i+1))), "Selected Action:", font=self.font40, fill=(255,255,255,255))
        draw.text((padding_right, 100 + int((bar_height+10) * (i+2))), self.actions[action], font=self.font40, fill=(255,255,255,255))
        frame = np.array(img_pil)

        return frame

