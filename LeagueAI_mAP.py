#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

# This script us used to compute the detection precision mAP of a model against a test dataset

from LeagueAI_helper import input_output, LeagueAIFramework, detection
import time
import cv2
from os import listdir
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def get_label_box(f):
    with open(f) as label_file:
        objects = label_file.readlines()
    objects = [o.rstrip('\n') for o in objects]
    return objects

def load_classes(names_file):
    f = open(names_file, "r")
    names = f.read().split("\n")[:-1]
    return names

def compute_map(label_object, label_class, objects, w_in, h_in):
    # Set up the label box
    l_x1, l_y1, l_x2, l_y2 = w_in*(float(label_object[0])-float(label_object[2])/2), h_in*(float(label_object[1])-float(label_object[3])/2), w_in*(float(label_object[0])+float(label_object[2])/2), h_in*(float(label_object[1])+float(label_object[3])/2)
    #print("l_x1: {} l_y1: {} l_x2: {} l_y2: {}".format(l_x1, l_y1, l_x2, l_y2))
    iou = []
    best_match_obect = []
    # Find the best matching objects
    for o in objects:
        o_x1, o_y1, o_x2, o_y2 = o.x_min, o.y_min, o.x_max, o.y_max
        #print("o_x1: {} o_y1: {} o_x2: {} o_y2: {}".format(o_x1, o_y1, o_x2, o_y2))
        # If its not the same object, skip
        if int(label_class) != int(o.object_class):
            continue
        # Get the coordinates of the intersetcion rectangle
        inter_rect_x1 = max(o_x1, l_x1)
        inter_rect_y1 = max(o_y1, l_y1)
        inter_rect_x2 = min(o_x2, l_x2)
        inter_rect_y2 = min(o_y2, l_y2)
        #print("x1: {}, y1: {}, x2: {}, y2: {}".format(inter_rect_x1,inter_rect_y1,inter_rect_x2,inter_rect_y2))
        # Intersection area
        inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
        #print("inter_area: ", inter_area)
        # Union Area
        o_area = (o_x2 - o_x1 + 1) * (o_y2 - o_y1 + 1)
        l_area = (l_x2 - l_x1 + 1) * (l_y2 - l_y1 + 1)
        #print("o: {}, l: {}".format(o_area, l_area)) 
        iou.append(inter_area / (o_area + l_area - inter_area))
        best_match_obect.append(detection(o.object_class, o_x1, o_y1, o_x2, o_y2))
    if len(iou) > 0:
        return float(max(iou)), best_match_obect[iou.index(max(iou))]
    else:
        return 0, detection(-1,0,0,0,0)

new_model = False

image_folder = "test_map/images_hand/"
if new_model:
    name_file = "/home/oli/Workspace/LeagueAI/cfg/LeagueAI.names"
    label_folder = "test_map/labels_hand_5/"
else:
    name_file = "/home/oli/Workspace/LeagueAI/cfg/LeagueAI_2017.names"
    label_folder = "test_map/labels_hand_3/"
names = load_classes(name_file)
output_size = (int(1920/2), int(1080/2))
classes_number = len(names)
mAP_threshold = 0.5

if new_model:
    LeagueAI = LeagueAIFramework(config_file="cfg/LeagueAI.cfg", weights="weights/05_02_synthetic_LeagueAI/LeagueAI_final.weights", names_file="cfg/LeagueAI.names", classes_number = 5, resolution=int(960/1.5), threshold = 0.35, cuda=True, draw_boxes=True)
else:
    LeagueAI = LeagueAIFramework(config_file="cfg/LeagueAI_combined.cfg", weights="weights/LeagueAI_combined/LeagueAI_combined3.weights", names_file="cfg/LeagueAI.names", classes_number = 3, resolution=int(960), threshold = 0.35, cuda=True, draw_boxes=True)
    #LeagueAI = LeagueAIFramework(config_file="cfg/LeagueAI_2017.cfg", weights="weights/LeagueAI_2017_final.weights", names_file="cfg/LeagueAI_2017.names", classes_number = 3, resolution=640, threshold = 0.35, cuda=True, draw_boxes=True)

files = sorted(listdir(image_folder))
# Show the invidivual images
show_images = True

save_image = False
output_path = "test_map/mAP_images"
filename = "new_"

mAP_avg = [0] * classes_number
mAP_not_detected = [0] * classes_number
mAP_wrong_detected = [0] *classes_number
mAP_ground_truth = [0] * classes_number
font_size = 1
for i, f in enumerate(files):
    # Get the current frame from the image
    img = Image.open(image_folder+f)
    img = img.convert("RGB")
    R, G, B = img.split()
    img = Image.merge("RGB", [B, G, R])
    w, h = img.size
    frame = np.array(img)

    # Get the list of detected objects and their positions
    objects = LeagueAI.get_objects(frame)
    label_boxes = get_label_box(label_folder+f.split(".")[0]+".txt")
    print("{} Objects out of {} detected!".format(len(objects), len(label_boxes)))

    for labels in label_boxes:
        label_class = int(labels.split(" ")[0])
        label_object = labels.split(" ")[1:]
        mAP, best_detection = compute_map(label_object, label_class, objects, w, h)

        label_object = [w*(float(label_object[0])-float(label_object[2])/2), h*(float(label_object[1])-float(label_object[3])/2), w*(float(label_object[0])+float(label_object[2])/2), h*(float(label_object[1])+float(label_object[3])/2)]
        if best_detection.object_class == -1:
            mAP_not_detected[label_class] += 1
            mAP_ground_truth[label_class] += 1
        else:
            if mAP >= mAP_threshold:
                mAP_avg[int(best_detection.object_class)] += 1
            else:
                mAP_wrong_detected[label_class] += 1
            mAP_ground_truth[label_class] += 1
            # Legend
            t_size = cv2.getTextSize("Detection", cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]
            cv2.rectangle(frame, (0, 0), (t_size[0], t_size[1]), (0,0,255), -1)
            cv2.putText(frame, "Detection", (0, int(t_size[1])), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 2)
            t_size = cv2.getTextSize("Label Ground Truth", cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]
            cv2.rectangle(frame, (0, t_size[1]), (t_size[0], 2*t_size[1]), (255,0,0), -1)
            cv2.putText(frame, "Label Ground Truth", (0, 2*int(t_size[1])), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 2)

            # Paint the boxes of label vs detection to visualize how the mAP is computed
            # Ground Truth
            cv2.rectangle(frame, (int(label_object[0]), int(label_object[1])), (int(label_object[2]), int(label_object[3])), (255,0,0), 2)
            # Detection
            cv2.rectangle(frame, (best_detection.x_min, best_detection.y_min), (best_detection.x_max, best_detection.y_max), (0,0,255), 2)
            t_size = cv2.getTextSize("iou: {}".format(round(mAP,3)), cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]

            # Print the IOU on the object
            cv2.rectangle(frame, (best_detection.x - int(t_size[0]/2), best_detection.y - int(t_size[1]/2)), (best_detection.x + int(t_size[0]/2), best_detection.y + int(t_size[1]/2)), (255,0,0),-1)
            cv2.putText(frame, "iou: {}".format(round(mAP,3)), (best_detection.x - int(t_size[0]/2), best_detection.y + int(t_size[1]/2)), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 2)

    if show_images:
        if save_image:
            print("saving to " +output_path+"/"+filename+str(i)+".jpg")
            print(cv2.imwrite(output_path+"/"+filename+str(i)+".jpg" , frame))
        while True:
            # Show the current image
            frame = cv2.resize(frame, output_size)
            cv2.imshow('LeagueAI', frame)
            if (cv2.waitKey(25) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break
    temp = [0] * classes_number
    for t in range(0, len(mAP_avg)):
        if mAP_ground_truth[t] > 0:
            temp[t] = mAP_avg[t]/mAP_ground_truth[t]
        else:
            temp[t] = 0
    print("mAP: ", temp)
    print("{} of {} images done!".format(i+1, len(files)))

#temp, mAP_avg, mAP_not_detected, mAP_wrong_detected, mAP_ground_truth = zip(*sorted(zip(temp, mAP_avg, mAP_not_detected, mAP_wrong_detected, mAP_ground_truth)))


print("")
print("Done")
print("correct detected:    ", mAP_avg)
print("not detected:        ", mAP_not_detected)
print("wrong detected:      ", mAP_wrong_detected)
print("ground truth :       ", mAP_ground_truth)
print("mAP:                 ", temp)

# Plot the resulting mAPs for each class
index = np.arange(classes_number)

fig = plt.figure()
fig.add_subplot(2,1,1)
width = 0.35

mAP_avg = np.array(mAP_avg)
mAP_not_detected = np.array(mAP_not_detected)
mAP_wrong_detected = np.array(mAP_wrong_detected)
p1 = plt.bar(index, mAP_avg, width=width)
p2 = plt.bar(index, mAP_not_detected, width=width, bottom=mAP_avg)
p3 = plt.bar(index, mAP_wrong_detected, width=width, bottom=[mAP_avg[i] + mAP_not_detected[i] for i in range(0, classes_number)])

plt.xticks(index, names)
plt.legend((p1[0], p2[0], p3[0]), ('True positives with mAP threshold = {}'.format(mAP_threshold), 'Not detected', 'Wrong detected'))
plt.xlabel("Classes")
plt.ylabel("Number of objects in the test set")

fig.add_subplot(2,1,2)
plt.bar(index, temp, width = 0.35)
plt.xticks(index, names)
plt.xlabel("Classes")
plt.ylabel("mAP")
plt.ylim([0.0,1.0])
plt.show()

