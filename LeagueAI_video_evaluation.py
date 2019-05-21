#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

# This file was used to evaluate the tracking performance of the player character in a evaluation video clip

from LeagueAI_helper import input_output, LeagueAIFramework, detection
import time
import cv2

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
IO = input_output(input_mode='videofile', video_filename='videos/eval.mp4')
####################

LeagueAI = LeagueAIFramework(config_file="cfg/LeagueAI.cfg", weights="weights/05_02_synthetic_LeagueAI/LeagueAI_final.weights", names_file="cfg/LeagueAI.names", classes_number = 5, resolution=int(960/1.5), threshold = 0.75, cuda=True, draw_boxes=True)
#LeagueAI = LeagueAIFramework(config_file="cfg/LeagueAI_combined.cfg", weights="weights/LeagueAI_combined/LeagueAI_combined3.weights", names_file="cfg/LeagueAI_2017.names", classes_number = 3, resolution=960, threshold = 0.75, cuda=True, draw_boxes=True)

v_at_least_1 = 0
v_exactly_1 = 0
v_none = 0
total = 0
while True:
    start_time = time.time()
    # Get the current frame from either a video, a desktop region or webcam (for whatever reason)
    frame = IO.get_pixels()
    # Get the list of detected objects and their positions
    objects = LeagueAI.get_objects(frame)
    vayne = 0
    for o in objects:
        #o.toString()
        if o.object_class == 2:
            vayne += 1
        # Example to validate if the returned objects are correct:
        # Draw rectangles arount the objects and a center point, you can use the draw_boxes=False to turn the frameworks boxes off
        #cv2.circle(frame, (o.x, o.y), 5, (255,0,0), -1)
        #cv2.rectangle(frame, (o.x_min, o.y_min), (o.x_max, o.y_max), (255, 0, 0), 4)
    if vayne == 1:
        v_at_least_1 += 1
        v_exactly_1 += 1
    elif vayne > 1:
        v_at_least_1 += 1
    else:
        v_none += 1
    total += 1
    print("exactly1: ", round(v_exactly_1/total,4), " at_least1: ", round(v_at_least_1/total,4), " none: ", round(v_none/total,4))


    """
    Here you can do stuff with the detected objects and the other data from above.
    This file just provides the minimal working version on which you can start building your own bots.
    If you need some inspiration check out my Vayne bot which should be included in the repository.
    Lets build something cool!
    """

    # Write fps
    cycle_time = time.time()-start_time
    cv2.putText(frame, "FPS: {}".format(str(round(1/cycle_time,2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    # Show the AI view, rescale to output size
    if show_window:
        frame = cv2.resize(frame, output_size)
        cv2.imshow('LeagueAI', frame)
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break



