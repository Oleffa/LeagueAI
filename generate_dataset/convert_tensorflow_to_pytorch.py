#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

# This script converts the XML annotations for a Tensorflow dataset to a txt annotation used by my Pytorch implementation

import xml.etree.ElementTree as ET
from os import listdir

input_directory = "/home/oli/Workspace/LeagueAI/generate_dataset/2017_LeagueAIDataset/labels_xml"
names_file = "/home/oli/Workspace/LeagueAI/cfg/LeagueAI.names"
output_directory = "/home/oli/Workspace/LeagueAI/generate_dataset/2017_LeagueAIDataset/labels_test"

# Generate the list of possible classes
# Or hardcode it
def load_classes(names_file):
    f = open(names_file, "r")
    names = f.read().split("\n")[:-1]
    return names
names = load_classes(names_file)
names = ['Tower', 'EnemyMinion', 'Vayne']

# List of all xml files in the directory
files = sorted(listdir(input_directory))

for f in files:
    if str(f.split('.')[-1]) != 'xml':
        print("skipping file")
    else:
        # Get root of the xml tree
        root = ET.parse(input_directory+"/"+f).getroot()
        # read the filename of the corresponding jpg
        filename = root.find('filename').text
        # Read all objects in the tree
        objects = root.findall('object')
        # Get the size of the image
        image_size = float(root.find('size').find('width').text), float(root.find('size').find('height').text)
        for o in objects:
            # Determine the class of the object
            object_class = o.find('name').text
            object_class = names.index(object_class)
            #Determine the positions of the object
            xmin = int(o.find('bndbox').find('xmin').text)
            ymin = int(o.find('bndbox').find('ymin').text)
            xmax = int(o.find('bndbox').find('xmax').text)
            ymax = int(o.find('bndbox').find('ymax').text)
            print("xmin: {}, y_min: {}, x_min: {}, x_max: {}".format(xmin, ymin, xmax, ymax))
            # Compute height, width and the center coordinates relative to the image size
            w = (xmax - xmin)/image_size[0]
            h = (ymax - ymin)/image_size[1]
            center_x = (xmin+(w*image_size[0]/2))/image_size[0]
            center_y = (ymin+(h*image_size[1]/2))/image_size[1]
            output_file = output_directory+"/"+f.split('.')[0]+".txt"
            print("center_x: {}, center_y: {}, w: {}, h: {}".format(center_x, center_y, w, h))
            with open(output_file, "a") as x:
                x.write("{} {} {} {} {}\n".format(object_class, center_x, center_y, w, h))


