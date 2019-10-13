#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

import sys
sys.path.append('..')
from LeagueAI_helper import input_output, LeagueAIFramework
from localization.posecells import *
import time
import cv2
import agent_helper

####### Params ######
# Show the AI view or not:
show_window = True
# Output window size
screen_size = (1920,1080)
video_to_view_factor = 2
ai_view_resolution = int(screen_size[0]/video_to_view_factor), int(screen_size[1]/video_to_view_factor)
# To record the desktop use:
#IO = input_output(input_mode='desktop', SCREEN_WIDTH=1920, SCREEN_HEIGHT=1130, x=0, y=50)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
#IO = input_output(input_mode='videofile', video_filename='../videos/posecell_test.mp4')
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

# Skip frames of video
# TOOD move to IO
for i in range(0, 20):
    frame = IO.get_pixels(scale=screen_size)


# Posecell localization stuff
view_templates = ViewTemplates(vt_size_x=80, vt_size_y=50, vt_x_min=200,
                               vt_y_min=80, vt_x_max=1800, vt_y_max=1080,
                               rate=0, template_match_threshold=0.07)
pc = PosecellNetwork(pc_x_size=80, pc_y_size=80, pc_w_e_dim=9, pc_w_i_dim=9,
                     find_best_kernel_size = 3, vt_active_decay=0.5,
                     pc_vt_inject_energy=0.5, pc_vt_restore=0.04,
                     pc_w_e_var=1.0, pc_w_i_var = 1.0, pc_global_inhib=0.005,
                     init_x=4, init_y=80-5, scale_factor=5, pc_cells_average=2)

vo = None
speed, angle = 0, 0
draw_movement_vector = True # Agent needs to be active for it to work

# TODO
# - Actuators
# - Translate the xy click pos to absolute screen pos to click

# TODO
# why are we moving faster to east than to the other directions?
# create_experience(), do we need them at all?

# TODO
# implement the enemy recognition

while True:
    # ======= Image pipeline ======
    start_time = time.time()
    for i in range(0, 3):
        frame = IO.get_pixels(scale=screen_size)
    if vo is None:
        vo = VisualOdometry(frame, 5, draw=False)
        speed, angle = 0, 0
    else:
        speed, angle = vo.get_optical_flow(frame)
    if speed < 0.5:
        speed = 0.0
    # Localization using posecells
    match_id = view_templates.on_image(frame)
    pc.on_view_template(match_id)
    global_x, global_y = pc.update(speed, angle)
    #print(global_x, global_y)
    #pc.plot_posecell_network(fps, view_templates.memory,  global_x, global_y)
    pc_not_updated = True

    if agent_active:
        objects = LeagueAI.get_objects(frame)
        # ======= Vayne AI code =======
        # Update the player position
        player.find_player_character(4, objects, 0.95)
        if player.detected is True:
            # Compute player HP
            hp, frame = player.compute_hp(frame, draw_hp_box=True)
            frame = player.draw_bb(frame, (255, 0, 0))



            # Get the distance of each object to the player character
            objects_sorted = LeagueAI.get_objects_sorted(objects, available_objects)
            for object_class in objects_sorted:
                for o in object_class:
                    if o.object_class != 4:
                        o.distance = player.get_distance(o)
            # Find closest object of each class
            shortest_list = [] # Stores the closest objects for each class with the nth element corresp. to the nth class
            distances = [-1]*4
            for l in objects_sorted:
                cur_object_class = l[0].object_class

                if cur_object_class != 4:
                    closest, distance = player.get_shortest(l)
                    shortest_list.append(closest)
                    distances[cur_object_class] = distance
            # Visualize the objects and their distances
            for o in objects_sorted:
                for l in o:
                    if l.object_class != 4:
                        if l.distance <= 450:
                            color = (0, 0, 255)
                        elif l.distance > 450 and l.distance < 750:
                            color = (0, 255, 0)
                        else:
                            color = (255, 0, 0)
                        cv2.rectangle(frame, (l.x_min, l.y_min), (l.x_max, l.y_max), color, 2)
                        cv2.line(frame, (player.x, player.y), (l.x, l.y), color, thickness=2)
            # Decision making
            i = 0
            attack_tower_prob = 0
            attack_canon_prob = 0
            attack_caster_prob = 0
            attack_melee_prob = 0
            retreat_prob = 0
            push_prob = 0
            for l in shortest_list:
                if l.object_class == 0: # Tower
                    attack_tower_prob = player.attack_prob(i, l.distance, player.hp)
                elif l.object_class == 1: # Canon
                    attack_canon_prob = player.attack_prob(i, l.distance, player.hp)
                elif l.object_class == 2: # Caster
                    attack_caster_prob = player.attack_prob(i, l.distance, player.hp)
                elif l.object_class == 3: # Melee
                    attack_melee_prob = player.attack_prob(i, l.distance, player.hp)
                else:
                    # If no enemies are found move back to midlane and push
                    push_prob = 100
                i += 1
            retreat_prob = player.retreat_prob((max(0, min(distances))), player.hp)
            #print("atow: {}, acan: {}, acast: {}, amele: {}, push: {}, retreat: {}".format(attack_tower_prob, attack_canon_prob, attack_caster_prob, attack_melee_prob, push_prob, retreat_prob))
            action = player.decide_action(attack_tower_prob, attack_canon_prob, attack_caster_prob,
                                          attack_melee_prob, push_prob, retreat_prob)
            # Visualize the probabilites
            frame = player.show_probs(frame, attack_tower_prob, attack_canon_prob, attack_caster_prob,
                                      attack_melee_prob, push_prob, retreat_prob, action)

            # Execute the action and get the odometry
            """
            if player.actions[action] ==  'Attack Tower':
                pass
            elif player.actions[action] ==  'Attack Canon':
                # Left Click the closest Canon minion
                #player.click_xy(shortest_list, obj_class=1, button=0)
                pass
            elif player.actions[action] ==  'Attack Caster':
                pass
            elif player.actions[action] ==  'Attack Melee':
                pass
            elif player.actions[action] ==  'Pushing':
                # Move
                vtrans_x, vtrans_y = 
                pass
            elif player.actions[action] ==  'Retreating':
                pass
            """
    # ======= Draw the movement vector ==========
    if draw_movement_vector and agent_active:
        f = 80
        x1 = player.x
        y1 = player.y
        x2 = int(round(x1 - speed * f * np.cos(angle)))
        y2 = int(round(y1 - speed * f * np.sin(angle)))
        frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
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




