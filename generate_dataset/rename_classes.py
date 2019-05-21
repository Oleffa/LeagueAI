#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

from os import listdir

# This script was used because initially some classes in the label folders had wrong number representations
# Its not very flexeible but i decided to keep it around in case someone else
# needs to change the object class numbers after generating a dataset

labels_path = "/home/oli/Workspace/LeagueAI/generate_dataset/Dataset_3/labels"

for i in listdir(labels_path):
    print(i)
    with open(labels_path+"/"+i) as f:
        content = f.readlines()
    new_content = []
    for j in range(0, len(content)):
        cur_line = content[j]
        obj_class = cur_line.split(" ")[0]
        rest = [0] + cur_line.split(" ")[1:]
        if int(obj_class) == 6:
            obj_class = 1
        elif int(obj_class) == 7:
            obj_class = 2
        elif int(obj_class) == 8:
            obj_class = 3
        elif int(obj_class) == 14:
            obj_class = 4
        rest[0] = obj_class
        output_line = rest
        new_content.append(str(output_line[0]) + " " + str(output_line[1]) + " " + str(output_line[2]) + " " + str(output_line[3]) + " " + str(output_line[4]))
    with open(labels_path+"/"+i, "w") as f:
        f.writelines(new_content)

