# This script just needs Opencv to run
# Install opencv using pip: python -m pip install opencv-python
# See this link for more information: https://www.scivision.co/install-opencv-python-windows/
import cv2
import math
import os
import datetime


#======================= Get parameters =========================
print("Welcome to Frameexporter, a tool that reads from a video file and exports every X frames into JPG images")
SOURCE_FILENAME = input("Please enter the filename:\n")
try:
    SKIP_FRAMES = int(input("Enter number of frames to skip between each export (0 means every frame will be exported)\n"))+1
except Exception:
    SKIP_FRAMES = 1
    print("Nothing entered, saving all frames!")

OUT_FILE_PREFIX = input("Enter a file prefix (for example out-). Entering nothing will just number the output in ascending order.\n")
source = cv2.VideoCapture(SOURCE_FILENAME)
t = datetime.datetime.now()
timestamp = t.strftime('%Y_%m_%d_%H_%M_%S')
output_directory = os.path.dirname(os.path.realpath(__file__)) + '/' + timestamp + "-export/"
print("Creating new directory for output: {}".format(output_directory))
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
else:
    print("error")


#====================== Resize output =========================
resize = True
try:
    x_pixel = int(input("Enter the output resolution X-axis in pixels. None means native resolution.\n"))
except Exception:
    resize = False
    print("No resolution entered, using native resolution")
if resize:
    try:
        y_pixel = int(input("Enter the output resolution Y-axis in pixels. None means native resolution.\n"))
    except Exception:
        print("No y resolutoin entered, using native resolution")
        resize = False

#===============================================================
print("Starting export, cancel the process by pressing ctrl+c. All images that are already exported will be saved!")
if (source.isOpened() == False):
    print('Error opening video stream')
else:
    frame_count = 0
    output_counter = 0
    while(source.isOpened()):
        ret, frame = source.read()
        if ret:
            if frame_count % SKIP_FRAMES == 0: 
                #Write to file
                output_filename = output_directory + OUT_FILE_PREFIX + str(output_counter) + ".jpg"
                output_counter = output_counter + 1
                if resize:
                    frame = cv2.resize((frame), (x_pixel,y_pixel))
                cv2.imwrite(output_filename, frame)
            frame_count = frame_count + 1

        else:
            print('File Processed!')
            break
        

