#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

from os import listdir
import numpy as np
import random

# This script is used to check if a label exists for every jpeg, if not it creates an empty label in labels
# Should not be needed if bootstrap.py does not have any bugs

# Set number of datasets that will be randomly selected for test dataset
test_dataset_size = 500
# Overall size of the dataset
dataset_size = 3000

# Directory of the dataset (parent directory of jpegs and labels folder)
dataset_path = "/home/oli/Workspace/LeagueAI/generate_dataset/Dataset"

# Randomly shuffle the list of samples in the dataset and select random test and train samples
datasets = sorted(listdir(dataset_path+"/jpegs/"))
datasets_txt = sorted(listdir(dataset_path+"/labels/"))
print(len(datasets))
print(len(datasets_txt))

for i in range(0, len(datasets)):
    # this should exist
    filename_txt = dataset_path+"/labels/"+datasets[i].split(".")[0]+".txt"
    found = False
    for j in range(0, len(datasets_txt)):
        cur = dataset_path+"/labels/"+datasets_txt[j].split(".")[0]+".txt"
        if cur == filename_txt:
            found = True
            break
    if found is False:
        print("error, filename_txt not found: ", filename_txt)
        with open(filename_txt, "a") as f:
            f.write("")

