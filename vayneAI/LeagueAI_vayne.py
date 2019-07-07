#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

import sys
sys.path.append('..')
from LeagueAI_helper import input_output, LeagueAIFramework
import time
import cv2
import agent_helper
import numpy as np

####### Params ######
# Show the AI view or not:
show_window = True
# Output window size
screen_size = (3440, 1440)
video_to_view_factor = 2
ai_view_resolution = int(screen_size[0]/video_to_view_factor), int(screen_size[1]/video_to_view_factor)
# To record the desktop use:
#IO = input_output(input_mode='desktop', SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
IO = input_output(input_mode='videofile', video_filename='../videos/eval.mp4')
####################

LeagueAI = LeagueAIFramework(config_file="../cfg/LeagueAI.cfg", weights="../weights/05_02_LeagueAI/LeagueAI_final.weights",
                             names_file="../cfg/LeagueAI.names", classes_number = 5, resolution=int(960/1.5),
                             threshold = 0.85, cuda=True, draw_boxes=False)

player_object_class = 4
available_objects = [0,1,2,3,4]
player = agent_helper.Player(0,0,player_object_class)
# Colors for enemy tower, canon, caster , melee
class_colors = [(0, 0, 255), (71, 99, 255), (0, 165, 255), (0, 215, 255)]

# Manual Controls
fps = IO.get_fps()
agent_active = True

for i in range(0, 600):
    frame = IO.get_pixels(scale=screen_size)

while True:
    # ======= Image pipeline ======
    start_time = time.time()
    frame = IO.get_pixels(scale=screen_size)
    if agent_active:
        objects = LeagueAI.get_objects(frame)
        # ======= Vayne AI code =======
        # Update the player position
        player.find_player_character(4, objects, 0.95)
        if player.detected is True:
            # Compute player HP
            hp, frame = player.compute_hp(frame, draw_hp_box=True)
            frame = player.draw_bb(frame, (255, 0, 0))

        objects_sorted = LeagueAI.get_objects_sorted(objects, available_objects)
        # Find closest minion
        shortest_list = []
        distances = []
        for l in objects_sorted:
            cur_object_class = l[0][0].object_class
            if cur_object_class != 4:
                closest, distance = player.get_shortest(l[0])
                shortest_list.append(closest)
                distances.append(distance)
        # Visualize the closest objects
        i = 0
        for l in shortest_list:
            cv2.rectangle(frame, (l.x_min, l.y_min), (l.x_max, l.y_max), class_colors[l.object_class], 2)
            cv2.line(frame, (player.x, player.y), (l.x, l.y), class_colors[l.object_class], thickness=1)
        # Decision making
        attack_tower_prob = 0
        attack_canon_prob = 0
        attack_caster_prob = 0
        attack_melee_prob = 0
        retreat_prob = 0
        push_prob = 0
        if l.object_class == 0: # Tower
            attack_tower_prob = player.attack_prob(0, distances[i], player.hp)
        elif l.object_class == 1: # Canon
            attack_canon_prob = player.attack_prob(1, distances[i], player.hp)
        elif l.object_class == 2: # Caster
            attack_caster_prob = player.attack_prob(2, distances[i], player.hp)
        elif l.object_class == 3: # Melee
            attack_melee_prob = player.attack_prob(3, distances[i], player.hp)
        else:
            # If no enemies are found move back to midlane and push
            push_prob = 100
        i += 1
        retreat_prob = player.retreat_prob(min(distances), player.hp)
        action = player.decide_action(attack_tower_prob, attack_canon_prob, attack_caster_prob,
                                      attack_melee_prob, push_prob, retreat_prob)
        # Visualize the probabilites
        frame = player.show_probs(frame, attack_tower_prob, attack_canon_prob, attack_caster_prob,
                                  attack_melee_prob, push_prob, retreat_prob, action)
        # TODO
        # - only print line to the target that is being attacked (in red), that is closest when running (blue) also with same colorborder
        # - The other make their boxes colored based on how close they are (blue - red)
        # - Actuators
    # ======= Write fps ===========
    cycle_time = time.time()-start_time
    cv2.putText(frame, "FPS: {}".format(str(round(1/cycle_time,2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    waitkey_time = int(1000/fps)
    # Show the AI view, rescale to output size
    if show_window:
        frame = cv2.resize(frame, ai_view_resolution)
        cv2.imshow('LeagueAI', frame)
        if cv2.waitKey(waitkey_time) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        waitkey_time = 1
    if cv2.waitKey(waitkey_time) & 0xFF == ord('a'):
        print("a pressed")
        agent_active = not agent_active




