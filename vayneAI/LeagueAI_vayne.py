#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

import sys
sys.path.append('..')
from LeagueAI_helper import input_output, LeagueAIFramework, detection
import time
import cv2
import agent_helper
import numpy as np

####### Params ######
# Show the AI view or not:
show_window = True
# Output window size
output_size = int(3440/3), int(1440/3)
# To record the desktop use:
#IO = input_output(input_mode='desktop', SCREEN_WIDTH=1920, SCREEN_HEIGHT=1080)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
IO = input_output(input_mode='videofile', video_filename='../videos/eval.mp4')
####################

LeagueAI = LeagueAIFramework(config_file="../cfg/LeagueAI.cfg", weights="../weights/05_02_LeagueAI/LeagueAI_final.weights", names_file="../cfg/LeagueAI.names", classes_number = 5, resolution=int(960/1.5), threshold = 0.75, cuda=True, draw_boxes=False)

player = agent_helper.Player(0,0,4)

while True:
    # ======= Image pipeline ======
    start_time = time.time()
    frame = IO.get_pixels()
    objects = LeagueAI.get_objects(frame)
    # ======= Vayne AI code =======
    # Run HP detection independedt from tracking
    #agent_helper.get_player_hp(frame)
    
    # Update the player position if it is detected
    player_detection = agent_helper.find_player_character(4, objects, 0.95)
    if player_detection is not None:
        player.update(player_detection.x.item(), player_detection.y.item(), player_detection.object_class.item())

    # Draw the player bounding box
    TODO draw the bounding box properly
    frame = agent_helper.draw_bb(frame, player.x, player.y, player.w, player.h)


    # ======= Write fps ===========
    cycle_time = time.time()-start_time
    cv2.putText(frame, "FPS: {}".format(str(round(1/cycle_time,2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    # Show the AI view, rescale to output size
    if show_window:
        frame = cv2.resize(frame, output_size)
        cv2.imshow('LeagueAI', frame)
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break



