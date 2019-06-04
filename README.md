# LeagueAI
Implementation of an A.I. Player for the videogame League of Legends based on Image Recognition using PyTorch

Attention: The old version can be found in the branch "LeagueAI_2017".

Attention: This project is still under development, the datasets I generate and the YOLOv3 object detection weights I trained will be made available soon! Meanwhile check out the included report on how to generate synthetic training data for League of Legends.

# OLD VERSION
This is a newer version of the old implementation and shall provide a framework that lets you build your own applications based on detecting objects in the game. As an example I will implement the old 2017 LeagueAI bot on this framework.
Demo video of Tensorflow implementation from 2017: https://www.youtube.com/watch?v=KRWFCaXfOTk

## TODO
1) Dataset generation, add these objects to the raw data
- Fix some bugs with the tower cropping of the raw data
- Towers in fog of war
- Turret plating
- Fog of war strucutres, maybe fog of war filter?
- Dead minions
- Add random particles to the screenshots, explosions and so on
- More different cursors
- More cursors
- health bars

2) Other TODOs:
- Extract object positions from the minimap to get an understanding of the global map.
- The mAP calculation needs a rework, sometimes objects are used twice for map computation.


## Currently Detectable Objects
- Red Tower
- Red Canon Minion
- Red Caster Minion
- Red Melee Minion
- Vayne

## Missing Objects
- More champions
- Add Red inhibitors, nexus, super minions
- Add Blue towers, nexus, inhibitors, minions

## Abstract
The task is to create an agent that is able to play 3rd person massive multiplayer online battle arena games (MOBA) like League of Legends, Dota 2 and Heroes of the Storm with the same input as a human player, namely vision.
Since League of Legends does not provide an interface to the game, object detection is used.
In this project a python implementation of Yolo v3 object detector and a way of randomly generating an infinite amout of training data is introduced.

## Object detection
For more information on the object detector refer to my technical report at: https://arxiv.org/abs/1905.13546 or directly to the YOLOv3 website [2]

## Installation/Usage
TODO

## The LeagueAI Dataset
Creating large datasets from scratch can be very work intensive.
For the first implementation of the LeageAI about 700 hand labeled pictures were used.
Labeling 700 pictures took about 4 days of work and only included 4 game objects (1 champion model, allied and enemy minions and enemy towers).
Therefore, the new dataset was created by automatically generating training data based on 3D models extracted from the game.

1. Obtaining champion and minion models form 3D models
To obtain the image data I used the online League of Legends model viewer from https://teemo.gg/model-viewer.
For each ingame object and each animation I recorded a short video clip while rotating the 3D model.
Next I used the `pyFrameExporter.py` script to extract individual pictures from the clips.
For the minions I used Adobe After Effects to add a green background to the videos of the minions and towers (all objects where I could not find the 3D models).
For each of the objects exported frames I used the `pyExportTransparentPNG.py` script.
The script removes the green/purple background from the individual screenshots and leaves you with the masked png of an objects.
Furthermore, the scrip crops the images to the content and removes excess seethrough space.

This leaves me with about 1000 masked images of each object that can be later used to generate labeled fake screenshots of the game.

2. Combining the masked and cropped images with game background 
To generate a large amount of training data that cover all regions of the game map, I generated a series of 200 screenshots from all over the map using the frame exporter script.
Then the masked and cropped images from step 1 are randomly combined with the map screenshots using the bootstrap.py script.
Since the images are placed using a script it is possible to obtain the objects position in the image and thus automatically generate a label for it.

To generate a large variety of screenshots the script can be adjusted to:
- change the random amount of champions, minions and other objects 
- randomly add a number of cursers
- randomly add the game HUD to the screenshot
- randomly scale all the objects
- cluster the minions to create more realistic clumps of fighting minions
- apply gaussian blur and random noise to the image

Using this method a dataset of many thousands of different labeled fake screenshots can be generated in a matter of hours.

A full description of the dataset generation process and evaluation compared to hand labeled data can be found in the following publication: https://arxiv.org/pdf/1905.13546.pdf

To cite please use:
```
@article{leagueaidatasetgeneration,
  title={LeagueAI: Improving object detector performance and flexibility through automatically generated training data and domain randomization},
  author={Struckmeier, Oliver},
  journal = {arXiv},
  year={2019}
}
```

## Extracting health information
TODO


# Sources
[1] Implementing Yolov3 object detection from scratch: https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch

[2] For training: https://pjreddie.com/darknet/yolo/, Yolov3: An Incremental Improvement, J. Redmond and A. Farhadi, 2018
