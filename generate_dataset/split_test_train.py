#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

from os import listdir
import numpy as np
import random

# This script is used to split the datasets generated with bootstrap.py into test and training dataset

# Attention if you use darknet to train: the structure has to be exactly as follows:
# - Dataset
# -- images
# --- XYZ0.jpg
# --- XYZ1.jpg
# -- labels
# --- XYZ0.txt
# --- XYZ1.txt
# -- train.txt
# -- test.txt

# Set number of datasets that will be randomly selected for test dataset
test_dataset_size = 500
# Overall size of the dataset
dataset_size = 3000

# Directory of the dataset (parent directory of jpegs and labels folder)
dataset_path = "/home/oli/Workspace/LeagueAI/generate_dataset/Dataset"

# Randomly shuffle the list of samples in the dataset and select random test and train samples
datasets = sorted(listdir(dataset_path+"/images/"))
random.shuffle(datasets)
datasets_test = datasets[:test_dataset_size]
datasets_train = datasets[test_dataset_size:]

print("Attention! Make sure that the test/train.txt are either empty or you really want to append to them!")

# Write the absolute file paths to the train/test files
with open(dataset_path+"/test.txt", "a") as f:
    for i in range(0, len(datasets_test)):
        f.write(dataset_path+"/images/"+datasets_test[i]+"\n")
with open(dataset_path+"/train.txt", "a") as f:
   for i in range(0, len(datasets_train)):
       f.write(dataset_path+"/images/"+datasets_train[i]+"\n")
   

