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
screen_size = (1920, 1080)
video_to_view_factor = 2
ai_view_resolution = int(screen_size[0]/video_to_view_factor), int(screen_size[1]/video_to_view_factor)
# To record the desktop use:
#IO = input_output(input_mode='desktop', SCREEN_WIDTH=screen_size[0], SCREEN_HEIGHT=screen_size[1]-20, x=0, y=20)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
#IO = input_output(input_mode='videofile', video_filename='../videos/posecell_test.mp4')
#IO = input_output(input_mode='videofile', video_filename='../videos/eval.mp4')
IO = input_output(input_mode='videofile', video_filename='recordings/good1.mp4')

####################

LeagueAI = LeagueAIFramework(config_file="../cfg/LeagueAI.cfg", weights="../weights/05_02_LeagueAI/LeagueAI_final.weights",
                             names_file="../cfg/LeagueAI.names", classes_number = 5, resolution=int(960/1.5),
                             threshold = 0.90, cuda=True, draw_boxes=False)

# video recording
# INput recorded using windows tools
record_processed = True
if record_processed:
    output_recorder = cv2.VideoWriter('recordings/processed.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 33.38, screen_size)
    pc_recorder = cv2.VideoWriter('recordings/pc.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 33.38, (400, 400))
    vo_recorder = cv2.VideoWriter('recordings/vo.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 33.38, (int(screen_size[0]/4),int(screen_size[1]/4)))
else:
    pc_recorder = None
    vo_recorder = None

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
for i in range(0, 300):
    frame = IO.get_pixels(scale=screen_size)

visual_odom = True

# Posecell localization stuff
if visual_odom:
    view_templates = ViewTemplates(vt_size_x=80, vt_size_y=50, vt_x_min=200,
                                   vt_y_min=80, vt_x_max=1800, vt_y_max=1080,
                                   rate=0, template_match_threshold=0.07)
    pc = PosecellNetwork(pc_x_size=80, pc_y_size=80, pc_w_e_dim=9, pc_w_i_dim=9,
                         find_best_kernel_size = 3, vt_active_decay=0.5,
                         pc_vt_inject_energy=0.5, pc_vt_restore=0.04,
                         pc_w_e_var=1.0, pc_w_i_var = 1.0, pc_global_inhib=0.005,
                         init_x=36, init_y=43, scale_factor=5, pc_cells_average=2)
    "Init for normal game start: x, 75"
    vo = None
    speed, angle = 0, 0
    draw_movement_vector = True # Agent needs to be active for it to work
else:
    draw_movement_vector = False

# TODO
# 1. record video using win+g and the AI playing
# 2. let the video run again through the thing but have everything on this time and video 30 fps, turn visual_odom to true

# TODO code for recording also pc and vo
# Check if it maybe works together with the posecell network
# If not record video of the bot only playing. and replay with the posecells

# Improve localization with some kind of memory of the map and ground truth positions
# Improve the pathing by making it relax like the experience map

while True:
    # ======= Image pipeline ======
    start_time = time.time()
    #for i in range(0, 6): # Careful changing the number of skipped frames here. This is equivalent to scaling the speed of the agent
    frame = IO.get_pixels(scale=screen_size)

    if visual_odom:
        if vo is None:
            # 10 is a better scale value
            vo = VisualOdometry(frame, 4, draw=True, recorder=vo_recorder)
            speed, angle = 0, 0
        else:
            speed, angle = vo.get_optical_flow(frame)
        # Localization using posecells
        match_id = view_templates.on_image(frame)
        pc.on_view_template(match_id)
        global_x, global_y = pc.update(speed, angle)
        #print("Global pos: ", global_x, global_y)
        pc.plot_posecell_network(fps, view_templates.memory,  global_x, global_y, recorder=pc_recorder)
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

            """
            # Update the distance of each object to the player character
            # Note needed anymore
            objects_sorted = LeagueAI.get_objects_sorted(objects, available_objects)
            for object_class in objects_sorted:
                for o in object_class:
                    if o.object_class != 4:
                        o.distance = player.get_distance(o)
            """
            # Find closest object of each class
            closest_tower = None
            closest_overall = None
            for o in objects:
                if o.object_class != 4:
                    o.distance = player.get_distance(o)
                    if closest_overall is None:
                        closest_overall = o
                    else:
                        if o.distance < closest_overall.distance:
                            closest_overall = o
                if o.object_class == 0:
                    o.distance = player.get_distance(o)
                    if closest_tower is not None:
                        if o.distance < closest_tower.distance:
                            closest_tower = o
                    else:
                        closest_tower = o

            # Visualize the objects and their distances
            kite_range = 220
            approach_range = 550
            tower_danger = 600
            for o in objects:
                if o.object_class != 4 and o.object_class != 0:
                    if o.distance <= 320:
                        color = (0, 0, 255)
                    elif o.distance > 320 and o.distance < 550:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    cv2.rectangle(frame, (o.x_min, o.y_min), (o.x_max, o.y_max), color, 2)
                    cv2.line(frame, (player.x, player.y), (o.x, o.y), color, thickness=2)
                elif o.object_class == 0:
                    if o.distance <= 600:
                        color = (255, 0, 0)
                        cv2.rectangle(frame, (o.x_min, o.y_min), (o.x_max, o.y_max), color, 2)
                        cv2.line(frame, (player.x, player.y), (o.x, o.y), color, thickness=2)
                    else:
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (o.x_min, o.y_min), (o.x_max, o.y_max), color, 2)
                        cv2.line(frame, (player.x, player.y), (o.x, o.y), color, thickness=2)
            """
            # Decision making
            # Are there enemies?
            if closest_overall is not None:
                print(closest_overall.distance)
                if closest_tower is not None:
                    # Tower behavior (keep distance to tower)
                    if player.get_distance(closest_tower) < tower_danger:
                        player.kite_away_from(closest_overall)
                    else:
                        # Normal routine
                        if player.get_distance(closest_overall) < kite_range:
                            player.kite_away_from(closest_overall)
                        elif player.get_distance(closest_overall) < approach_range:
                            player.attack(closest_overall)
                        else:
                            player.move_towards(closest_overall)
                else:
                    if player.get_distance(closest_overall) < kite_range:
                        player.kite_away_from(closest_overall)
                    elif player.get_distance(closest_overall) < approach_range:
                        player.attack(closest_overall)
                    else:
                        player.move_towards(closest_overall)
            else:
                player.push_mid()
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
    if not record_processed:
        cv2.putText(frame, "FPS: {}".format(str(round(1/cycle_time,2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    waitkey_time = int(1000/fps)
    # Show the AI view, rescale to output size
    if show_window:
        if record_processed:
            print(frame.shape)
            output_recorder.write(frame)
        frame = cv2.resize(frame, ai_view_resolution)
        cv2.imshow('LeagueAI', frame)
        if cv2.waitKey(waitkey_time) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        waitkey_time = 1
    if cv2.waitKey(waitkey_time) & 0xFF == ord('a'):
        print("a pressed")
        agent_active = not agent_active




