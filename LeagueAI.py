
# coding: utf-8

# # Imports and environment setup

# In[ ]:


import matplotlib
#matplotlib.use("Qt5Agg")
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import time
from mss import mss
import math
from math import exp
from random import randint
from random import random
from numpy.linalg import inv



#mouse control and window dimensions
SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 768
MONITOR_WIDTH = 800
MONITOR_HEIGHT = 600
import win32api, win32con
#Player Control
click_cooldown = 0.5

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util


# # Load Model

# In[ ]:


#Select object detection model from the same folder as this script
MODEL_NAME = 'LeagueAI_v3'
MODEL_FILE = MODEL_NAME + '.tar.gz'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'LeagueAI_v2.pbtxt')
# Number of classes in the pbtxt file that can be detected by the model
NUM_CLASSES = 3


# ## Load the frozen model into memory

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# Helper classes for various champion functions and interaction with the game

# In[ ]:


# click a postion in the screen using x,y coordinates in pixels
def click(x,y, attack):
    win32api.SetCursorPos((x,y))
    if attack == True:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    else:
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
# move the mouse cursor to x,y coordinates in pixels
def move_cursor_to(x,y):
    win32api.SetCursorPos((x,y))
# find the player character position marked as x,y (converting the boxes to a single point position)
# some hard coded things are done here: the character model center is not in the center of the detection
# but in the lower part near its feet. Thats why we determined some factors for calculating the position
# from the box experimentally
def find_box_xy(box):
    playerwidth =  (box[3] - box[1])
    playerheight = box[2] - box[0]
    playerpos_y = (box[0] + playerheight/1.3)
    playerpos_x = (box[1] + playerwidth/2)
    return([playerpos_y, playerpos_x, playerheight, playerwidth])
# find out in which grid zell on object is
def find_object_state(grid, minion_origin, player_origin, grid_width, grid_height):
    x_object = (minion_origin[0])#/MONITOR_WIDTH)#*SCREEN_WIDTH
    y_object = (minion_origin[1])#/MONITOR_HEIGHT)#*SCREEN_HEIGHT
    x_player = (player_origin[0])#/MONITOR_WIDTH)#*SCREEN_WIDTH
    y_player = (player_origin[1])#/MONITOR_HEIGHT)#*SCREEN_HEIGHT
    #find difference to player in pixels
    x_dif = -1*(x_player - x_object)
    y_dif = (y_player - y_object)    
    #transform pixels into states
    state_x = round(x_dif/grid_width)
    state_y = round(y_dif/grid_height)
    return([state_x, state_y])
# click a certain grid cell given by x and y coordinate in the state grid
# the click will click in the center of the grid
def click_state(state_x, state_y, state_width, state_height, player_origin, attack):
    x_click = (player_origin[0]/MONITOR_WIDTH)*SCREEN_WIDTH + state_x*(state_width/MONITOR_WIDTH)*SCREEN_WIDTH
    y_click = (player_origin[1]/MONITOR_HEIGHT)*SCREEN_HEIGHT + (-1*state_y*(state_height/MONITOR_HEIGHT)*SCREEN_HEIGHT)
    click(int(x_click), int(y_click), attack)
    #move_cursor_to(int(x_click), int(y_click))
# set a certain state in the grid to a certain value to mark for example the type of unit in the grid cell
def set_array_pos(grid, x, y_in, value, x_grid_size_in, y_grid_size_in):
    y = y_in*(-1)
    # make sure that we do not overwrite the state of our player character in the grid
    if not value == 1 and x == 0 and y == 0:
        return grid
    # resize grid in case we got uneven grid sizes
    if x_grid_size_in%2==1:
        x_grid_size = x_grid_size_in - 1
    else:
        x_grid_size = x_grid_size_in
    if y_grid_size_in%2==1:
        y_grid_size = y_grid_size_in -1
    else:
        y_grid_size = y_grid_size_in
    x_pos = x + int(x_grid_size/2)
    y_pos = y + int(y_grid_size/2)
    grid[x_pos][y_pos] = value    
    return grid

# the teleport and recall functions dont work because it is not possible to send button presses and klicks
# on the hud to the game
# the win32api does work on another driver level than the game

def teleport_to(screen_x, screen_y):
    #press shift+s and klick a coordinate
    win32api.keybd_event((0x10),0,0,0)
    win32api.keybd_event((0x10),0 ,win32con.KEYEVENTF_KEYUP ,0)
    time.sleep(0.05)
    win32api.keybd_event((0x53),0,0,0)
    win32api.keybd_event((0x53),0 ,win32con.KEYEVENTF_KEYUP ,0)
    time.sleep(0.05)
    #
    #
    win32api.SetCursorPos((screen_x,screen_y))
    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN,screen_x,screen_y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP,screen_x,screen_y,0,0)
def recall():
    win32api.SetCursorPos((929,720))
    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN,920,720,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP,920,720,0,0)
    #click(920,728 ,True)


# In[ ]:


class Policy:
    # The policy cointains the parameters used for making decisions and learning
    # More on the paramters can be found int the pdf report in chapter 4
    c_max_visible_dist = 20.0
    c_close_dist_cutoff = 1.0 / 10.0
    c_tower_safe_dist_cutoff = c_max_visible_dist / 3
    c_max_hp = 1
    # this parameter sets the time frame the bot should leave the base and go to lane
    # this paramtere requires some further testing. the bot is not really smart enough yet to know where to go
    # when the game starts. thats why the move to lane is hard coded and does not use this parameter for now
    c_game_time_beginning_cutoff = 0  # set time in seconds until the normal game routine starts  #0.5 *60 / 5
    policy_gradient = np.array([1, 1, 1, 1]).reshape( -1 )
    delta_theta = []
    delta_R = []
    R_i = 1
    alpha = 0.1
    R_ref = 1
    
    logistic_func_x_scale = 6
    
    # theta is the policy parameter we learn for the different actions:
    # atk minion, atk tower, move to goal, retreat
    theta = np.zeros( (4,1) )
    #other initial values for the parameters
    #theta[0] = c_close_dist_cutoff
    #theta[1] = 1
    #theta[2] = 0.8 / (35*60/5)
    #theta[3] = 0.2
    theta[0] = 1
    theta[1] = 1
    theta[2] = 1
    theta[3] = 1
class State:
    # This class defines the state in which the player is currently
    hp = 1.0
    cloest_minion_dist = 0
    tower_dist = 0
    # game time counted in iterations or steps
    start_time = time.time()
    game_time = time.time() - start_time


# More on how the learning and decision making works can be found in the pdf report in the repository in chapter 4.5

# In[ ]:


## Intelligence
def find_shortest_distance(unit_grid, unit_type):
    distance = 10000 #no distance found
    smallest_x = 0
    smallest_y = 0
    for i in range(3,len(unit_grid)-3):
     for j in range(3,len(unit_grid[i])-3):
        if unit_grid[i][j] == unit_type:#if minion found
            cur_dist=math.sqrt(math.pow(i-int(len(unit_grid)/2),2)+math.pow(j-int(len(unit_grid[0])/2),2))
            if cur_dist < distance:
                distance = cur_dist
                smallest_x = i-int(len(unit_grid)/2)
                smallest_y = j-int(len(unit_grid[0])/2)
    return [distance,smallest_x,-1*smallest_y]

def minion_probability(closest_minion_distance,hp_frac):
    #x: input to function
    #1: max output value
    #m: slope, calculated based on the max range in which an enemy is visible
    #max_dist_enemy: maximum distance in which an enemy is
    #x = closest_enemy_distance-closest_enemy_distance/5 #factor to increase minion threat level
    #y = 7.954*math.pow(10,-5)*math.pow(x,4)-0.003558*math.pow(x,3)+0.05299*math.pow(x,2)-0.3257*x+0.9511
    #print("closest_miniont: " + str(closest_minion_distance))
    #p = policy.theta[0]*(hp_frac/closest_minion_distance)
    
    p = policy.theta[0]*hp_frac*(1/( 1+exp( -2*(closest_minion_distance-3)) ) - 1/( 1+exp( -3*(closest_minion_distance-6)) ) )
    if   p > 1: return 1
    elif p < 0: return 0
    else      : return p

def tower_probability(closest_tower_distance, hp_frac):
    #y = 0.05657865 + (1.01699 - 0.05657865)/(1 + math.pow((closest_tower/6.996663),5.975224))
    #print("closest_tower: " + str(closest_tower_distance))
    
    # reposition the curve the centre around [-6, +6]
    #closest_tower_distance = closest_tower_distance - 0.17    # center the curve the offset or cutoff dist
    #closest_tower_distance = closest_tower_distance - 0.5     # scale the range from  [0, 1] (or actually [-0.17, 1-0.17])
    #closest_tower_distance = closest_tower_distance * 6*2
    #print("closest_tower[-6, +6]: " + str( (closest_tower_distance) ))
    p = policy.theta[1]*hp_frac*(1/( 1+exp( -2*(closest_tower_distance+2)) ) - 1/( 1+exp( -2*(closest_tower_distance-4.6)) ) )
    if   p > 1: return 1
    elif p < 0: return 0
    else      : return p

def goal_probability():
    #depending on game time. we assume that after a 60 min game the probabilty to approach is very high
    #print("game_time_beginning_cutoff remaining:" + str(state.game_time))
    if state.game_time < policy.c_game_time_beginning_cutoff:
        return 1
    else:
        #print("theta2: " + str(policy.theta[2][0])+ " game time: " + str(state.game_time))
        #p = policy.theta[2] * state.game_time
        p = policy.theta[2]*0.0002228777*state.game_time + 0.204687
        if   p > 1: return 1
        elif p < 0: return 0
        else      : return p
        
def retreat_probability(closest_enemy,hp_frac):
    #p = exp( policy.theta[3] / hp_frac ) - 1   
    #p = 0.9686364 - 1.217273*hp_frac + math.pow(0.3090909*hp_frac,2)
    x = hp_frac * policy.theta[3]
    x2 = closest_enemy * policy.theta[3]
    p = 0.94 - 1.26*hp_frac + 0.4*math.pow(hp_frac,2) + (-0.11*x2+0.7)
    if   p > 1: return 1
    elif p < 0: return 0
    else      : return p


def decide_action(minion_prob, tower_prob, goal_prob, retreat_prob):
   total_prob = minion_prob + tower_prob + goal_prob + retreat_prob

   attack_minion_cutoff = minion_prob / total_prob
   attack_tower_cutoff = (minion_prob + tower_prob) / total_prob
   goal_cutoff = (minion_prob + tower_prob + goal_prob) / total_prob
   retreat_cutoff = retreat_prob / total_prob

   #roll random number and decide for one of the actions
   rnd = random()
   action = 0
   if   rnd < attack_minion_cutoff: action = 0
   elif rnd < attack_tower_cutoff:  action = 1
   elif rnd < goal_cutoff:          action = 2
   else:                            action = 3
   return action


def updateR_i(delta_theta_i):
    R_i = policy.R_i
    alpha = policy.alpha
    
    R_i = (1-alpha)*policy.R_i + alpha*np.dot(policy.policy_gradient, delta_theta_i)
    policy.R_i = R_i
    
    return R_i

def updateR_ref(reward):
    R_ref = policy.R_ref
    alpha = policy.alpha
    
    R_ref = (1-alpha)*R_ref + alpha*reward
    policy.R_ref = R_ref
    
def estimate_policy_gradient_FD():
    #TODO do we need ticks here?
    #gets array with all actions in the last 5 seconds and the number of ticks/action done in the time frame
    #policy_grad = inv(policy.delta_theta.T * policy.delta_theta) * policy.delta_theta.T * policy.delta_R 
    delta_theta = np.array(policy.delta_theta)
    delta_R = np.array(policy.delta_R)
    
    policy_grad = ((inv((delta_theta.reshape((-1,4)).T).dot(delta_theta.reshape((-1,4))))
                  ).dot(delta_theta.reshape((-1,4)).T)).dot(delta_R.reshape((-1,1))).reshape(4)
    policy_grad_norm = np.linalg.norm(policy_grad.reshape(-1))
    policy_grad /= policy_grad_norm
    #print("pol_grad: " + str(policy_grad))
    #print("policy gradient shape")
    #print(policy_grad.shape)
    policy.policy_gradient = policy_grad

def perturbate_policy(delta_scale=1.0):
    factor = np.random.random(4)*delta_scale
    delta_theta_i = factor * np.array(policy.theta).reshape(4)
    return delta_theta_i.tolist()

    


# ## Main Routine

# In[ ]:


#=======move to lane code=====
# hard coded behaviour to move to the middle lane and wait until the game begins

#sct.get_pixels(mon)
#screen = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
#screen = np.array(screen)
#screen = cv2.resize(screen, (MONITOR_WIDTH, MONITOR_HEIGHT))
#image_np = screen
#cv2.imshow('AI_View', image_np)
#for i in range(1,14,1):
#    click(950,100,False)
#    print("moving to lane")
#    time.sleep(2)
#print("wait for battle to begin")
#time.sleep(55)
#================================
time.sleep(5)

# variable to track when we found the champion for the last time. so if the champion moves to a spot where it is
# not visible anymore it will move randomly until found again. or recall (if it would work ....)
vayne_last_found = time.time()
# Parameters for the rewards
MINION_REWARD = 10
TOWER_REWARD = -100
BLOCKED = -999
policy = Policy()
state = State()
# Initialize the rewards for 1 and 5 seconds average
reward_1 = 0
reward_5 = 0
state.start_time = time.time()

# game start stuff
start = False
start_time = time.time()

# reset variables for decision making and reward calculation
last_player_hp = 1
attack_number_1 = 0
attack_number_5 = 0
hp_change_1 = 0
hp_change_5 = 0
last_reset_1 = time.time()
last_reset_5 = time.time()
tick_count_1 = 0
tick_count_5 = 0
game_start = time.time()

# sct is used to get the pixels from the desktop
sct = mss()
# determine the size of the area to observe (the game)
# mon = {'top': 0, 'left': 0, 'width': SCREEN_WIDTH, 'height': SCREEN_HEIGHT} for fullscreen
mon = {'top': 0, 'left': 0, 'width': SCREEN_WIDTH, 'height': SCREEN_HEIGHT}
# timer to see how fast the loop is running
last_time = time.time()


#fill the delta theta with values or else there is an error
#delta_theta_i = perturbate_policy()
#policy.delta_theta.append(delta_theta_i)


print("theta0_minion;theta1_tower;theta2_approach;theta3_retreat")



with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      #print('=========Loop took: {} seconds'.format(time.time()-last_time))
      last_time = time.time()
      # update gametime
      state.game_time = time.time() - state.start_time
      # replaced by mss package
      #screen = cv2.resize(grab_screen(region=(0,40,1024,768)), (800,450))
      #screen = np.array(ImageGrab.grab(bbox=(0,40,1024,768)))
      # get screen recording and resize it to 800x450 pixels for output
      sct.get_pixels(mon)
      screen = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
      screen = np.array(screen)
      screen = cv2.resize(screen, (MONITOR_WIDTH, MONITOR_HEIGHT))
      image_np = screen
      # ===================tensorflow template code===============
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})                   
      # Visualization of the results of a detection
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=0.30,
          line_thickness=4)

      #===================Player Character=======================
      # detect hp:
      # read the number of green vs non green pixels from the health bar (hard coded for )
      hp_y_min = 572
      hp_y_max = hp_y_min + 8
      hp_x_min = int(MONITOR_WIDTH/2-115)
      hp_x_max = int(MONITOR_WIDTH/2+56)  
      # print the dimensions and positions of the square for debuging purpsoes
      #print(image_np[hp_y_min][hp_x_min])
      #print(image_np[hp_y_max][hp_x_max])
      #rectangle indicator to see the region where the hp are read from
      cv2.rectangle(image_np, (hp_x_min, hp_y_min),(hp_x_max,hp_y_max),(0, 0, 255), 1)
      green_pixels = 0
      for x_hp in range(hp_x_min,hp_x_max,1):
            for y_hp in range(hp_y_min,hp_y_max,1):
                if image_np[y_hp][x_hp][1] > 25:
                    green_pixels = green_pixels + 1
      #print("Player HP: " + str(green_pixels/((hp_x_max-hp_x_min-1)*(hp_y_max-hp_y_min-1))))
      playerHP = green_pixels/((hp_x_max-hp_x_min-1)*(hp_y_max-hp_y_min-1))
    
      # reset various game variables
      gameover = False # did the player die?
      vayne_found = False # did we found the player character?
      minion_count = 0 # how many enemy minions have been detected
      tower_count = 0 # how many towers are around?
      try:
          if vayne_last_found + 10 < time.time():
              # recall if the hap are low and then move back to lane
              # unfortunately the recall function is not working so we just click randomly until we find the
              # the player character again
              #if playerHP <= 0.15:
                  #recall to make sure we are in base then, doesnt work because i cant sent keystrokes or hud clicks ....
                  #recall()
                  #time.sleep(50)#wait death timerand move back to lane
                  #for i in range(1,14,1):
                  #    click(950,100,False)
                  #    print("moving to lane")
                  #    time.sleep(2)
                  #else:
              click_state(randint(-2,2),randint(-2,2),w,h,player_origin,False);
      except NameError:
          print('player_origin not set yet')
      for i,b in enumerate(boxes[0]):
          if scores[0][i] >= 0.3:
              #Vaynefound makes sure that the whole grid + decision proces only executed once per detection in case of a multi detection
              if classes[0][i] == 1 and vayne_found == False:
                  vayne_found = True;
                  vayne_last_found = time.time()
                  player_position = find_box_xy(boxes[0][i])
                  # draw a circle where the center of the player character is (not the center of the rectangle
                  # but some arbitrary spot between the legs of the model that was found experimentally)
                  player_origin = [int(player_position[1]*MONITOR_WIDTH), int(player_position[0]*MONITOR_HEIGHT)]
                  cv2.circle(image_np, (player_origin[0], player_origin[1]), 2, (0, 0, 255), 2)
                  #calculate the origin of the center rectangle under the character
                  origin = [int(player_origin[0]+9)-int(player_position[2]*MONITOR_WIDTH/3), player_origin[1]-15,int(player_origin[0])+20, int(player_origin[1])+18]        
                  cv2.rectangle(image_np,(origin[0], origin[1]),(origin[2], origin[3]), (0, 255, 0), 1)
                  #create draw states
                  w = int((origin[2] - origin[0]))
                  h = int((origin[3] - origin[1]))
                  grid_x = int(SCREEN_WIDTH/w)
                  grid_y = int(SCREEN_HEIGHT/h)
                  dim = (grid_x,grid_y)
                  unit_grid = np.zeros(dim) 
                  #set the origin of the grid to be the player character represented by a 1
                  #the grid goes from -grid_x/2 to grid_x/2 because the player is the center and everything left and under it is negative
                  unit_grid = set_array_pos(unit_grid,0,0,1,grid_x,grid_y)
                  #=======DETECT OBJECTS=======
                  #2.Minions
                  for j,k in enumerate(boxes[0]):
                      if scores[0][j] >= 0.3:
                          if classes[0][j] == 2:
                              minion_count = minion_count + 1
                              cur_minion_position = find_box_xy(boxes[0][j])
                              cur_minion_origin = [int(cur_minion_position[1]*MONITOR_WIDTH), int(cur_minion_position[0]*MONITOR_HEIGHT)]
                              cv2.circle(image_np, (cur_minion_origin[0], cur_minion_origin[1]), 2, (0, 0, 255), 2)
                              #Find in which state in the unit_grid the minion is
                              cur_minion_state = find_object_state(unit_grid, cur_minion_origin, player_origin, w, h)
                              unit_grid = set_array_pos(unit_grid,cur_minion_state[0],cur_minion_state[1],2,grid_x,grid_y)
                          #3.Towers
                          if classes[0][j] == 3:
                              tower_count = tower_count + 1
                              cur_tower_position = find_box_xy(boxes[0][j])
                              cur_tower_origin = [int(cur_tower_position[1]*MONITOR_WIDTH), int(cur_tower_position[0]*MONITOR_HEIGHT)]
                              cv2.circle(image_np, (cur_tower_origin[0], cur_tower_origin[1]), 2, (0, 0, 255), 4)
                              cur_tower_state = find_object_state(unit_grid, cur_tower_origin, player_origin, w, h)
                              unit_grid = set_array_pos(unit_grid,cur_tower_state[0],cur_tower_state[1],3,grid_x,grid_y)                 

                  #=============================
                  # Visualize Objects and generate reward matrix (each cell contains a certain reward)
                  # not really needed anymore but the code is still used to draw the objects in the grid
                  dim = (grid_x,grid_y)
                  reward_grid = np.zeros(dim)
                  for x in range(len(unit_grid)):
                      for y in range(len(unit_grid[0])):
                          #fill reward grid at the same time
                          if unit_grid[x][y] == 1:
                              #set the state at which the player is as blocked (-999 reward)
                              reward_grid[x][y] = BLOCKED
                              #===Paint player rectangle blue
                              cv2.rectangle(image_np,(origin[0]+w*x-int(grid_x/2)*w, origin[1]+h*y-int(grid_y/2)*h),(origin[2]+w*x-int(grid_x/2)*w, origin[3]+h*y-int(grid_y/2)*h), (0, 255, 0), 1)
                              cv2.rectangle(image_np,(origin[0]+2+w*x-int(grid_x/2)*w, origin[1]+2+h*y-int(grid_y/2)*h),(origin[2]-2+w*x-int(grid_x/2)*w, origin[3]-2+h*y-int(grid_y/2)*h), (0, 0, 255), 2)
                          elif unit_grid[x][y] == 2:
                              reward_grid[x][y] = MINION_REWARD
                              #===MINIONS
                              cv2.rectangle(image_np,(origin[0]+2+w*x-int(grid_x/2)*w, origin[1]+2+h*y-int(grid_y/2)*h),(origin[2]-2+w*x-int(grid_x/2)*w, origin[3]-2+h*y-int(grid_y/2)*h), (255, 0, 0), 2)
                          elif unit_grid[x][y] == 3:
                              reward_grid[x][y] = TOWER_REWARD
                              #===TOWER
                              cv2.rectangle(image_np,(origin[0]+2+w*x-int(grid_x/2)*w, origin[1]+2+h*y-int(grid_y/2)*h),(origin[2]-2+w*x-int(grid_x/2)*w, origin[3]-2+h*y-int(grid_y/2)*h), (255, 255, 255), 2)
                          else:
                              # draw empty rectangles green
                              cv2.rectangle(image_np,(origin[0]+w*x-int(grid_x/2)*w, origin[1]+h*y-int(grid_y/2)*h),(origin[2]+w*x-int(grid_x/2)*w, origin[3]+h*y-int(grid_y/2)*h), (0, 255, 0), 1)


                  #=======Hard Coded Decision Making/Outdated========
                  #if shortest_distance <= 5.0:
                  #      #print('too close! run!')
                  #      cv2.rectangle(image_np,(0,0), (w*5,h*5), (0, 255, 0), -1)
                  #      if closest_x > 0 and closest_y > 0:
                  #          click_state(-closest_x/abs(closest_x), -closest_y/abs(closest_y),w,h,player_origin,False)
                  #elif shortest_distance < 8.0 and shortest_distance > 5.0:
                  #      #print('attack!')
                  #      cv2.rectangle(image_np,(0,0), (w*3,h*3), (0, 0, 255), -1)
                  #      click_state(closest_x, closest_y,w,h,player_origin,True)
                  #      time.sleep(1)
                  #elif shortest_distance > 8.0 and minion_count > 0:
                  #      cv2.rectangle(image_np,(0,0), (w*3,h*3), (255, 0, 0), -1)
                  #      #print('not in range, approaching!')
                  #      click_state(closest_x, closest_y,w,h,player_origin,False)
                  #else:
                  #      cv2.rectangle(image_np,(0,0), (w*3,h*3), (0, 0, 0), -1)
                  #      #print('no enemies found, run towards enemy base!')
                  #      click_state(1, 1,w,h,player_origin,False)


                  #=======Calculate the threat values of objects==========
                  #get the info based on which the decisions shall be made
                  [shortest_distance_minion, closest_minion_x, closest_minion_y] = find_shortest_distance(unit_grid, 2)
                  [shortest_distance_tower, closest_tower_x, closest_tower_y] = find_shortest_distance(unit_grid, 3)
                  #shortest_distance_tower = shortest_distance_tower/math.sqrt(math.pow((grid_x/2),2)+math.pow((grid_y/2),2))
                  #print("closest minion" + str(shortest_distance_minion))
                  # calculate probabilites for each action
                  attack_tower_prob = tower_probability(shortest_distance_tower, playerHP)
                  approach_enemy_base_prob =  goal_probability()
                  retreat_prob = retreat_probability(shortest_distance_minion, playerHP)
                  if shortest_distance_minion == 10000: #if there are no minions
                      attack_minion_prob = 0
                      approach_enemy_base_prob = approach_enemy_base_prob + 0.1
                  else:
                      #shortest_distance_minion = shortest_distance_minion/math.sqrt(math.pow((grid_x/2),2)+math.pow((grid_y/2),2))
                      attack_minion_prob = minion_probability(shortest_distance_minion, playerHP)

                  
                  #print("min: " + str(attack_minion_prob) + " tow: " + str(attack_tower_prob) + " appr: " + str(approach_enemy_base_prob) + " retr: " + str(retreat_prob))
                  #print("attack_tower_prob: " + str(attack_tower_prob))
                  #print("approach_enemy_base_prob: " + str(approach_enemy_base_prob))
                  #print("retreat_prob: " + str(retreat_prob))

                  #=======Make decision which action to take and to which state====
                  #[state_x, state_y, action] = make_decision(playerHP, unit_grid, minion_value, tower_value);
                  action = decide_action(attack_minion_prob, attack_tower_prob, approach_enemy_base_prob, retreat_prob)
                  #print("selected action: " + str(action))
                  ##=======Execute the action===========
                  if action == 0:#Attack Minion
                      click_state(closest_minion_x, closest_minion_y, w, h, player_origin, True)
                      time.sleep(0.5)
                      attack_number_1 = attack_number_1 + 2
                      attack_number_5 = attack_number_5 + 2
                  elif action == 1:#Attack Tower
                      click_state(closest_tower_x, closest_tower_y, w, h, player_origin, True)
                      time.sleep(0.5)
                      attack_number_1 = attack_number_1 + 4
                      attack_number_5 = attack_number_5 + 4
                  elif action == 2:#Approach
                      #Just move to the top right side
                      click_state(1, 1, w, h, player_origin, False)
                  else: #run away from the closest enemy by just running the the oposite direction as the enemy minion is
                      if shortest_distance_minion == 10000 or closest_minion_x <= 0 or closest_minion_y <= 0:
                          click_state(-1, -1, w, h, player_origin, False)
                      else:
                          click_state(2*(-closest_minion_x/abs(closest_minion_x)), 2*(-closest_minion_y/abs(closest_minion_y)), w, h, player_origin, False);

                  #======Calculate Reward and feed it back======
                  # This part of the algorithm is roughly described in the pdf report in chapter 4.5
                  hp_change_1 = hp_change_1 + (- playerHP + last_player_hp) #add up the hp change until reset
                  hp_change_5 = hp_change_5 +  (- playerHP + last_player_hp)
                  tick_count_1 = tick_count_1 + 1
                  tick_count_5 = tick_count_5 + 1
                  last_player_hp = playerHP #update the current hp for the next loop
                  #other params: attack_number_1/5 number of attacks since last reset

                  cv2.rectangle(image_np,(0,0), (102,154), (255, 255, 255), 2)
                  #minion
                  cv2.rectangle(image_np,(2,2), (int(100*attack_minion_prob),30), (0, 0, 125), -1)
                  cv2.rectangle(image_np,(2,32), (int(100*attack_tower_prob),60), (0, 0, 255), -1)
                  cv2.rectangle(image_np,(2,62), (int(100*approach_enemy_base_prob),90), (255, 0, 0), -1)
                  cv2.rectangle(image_np,(2,92), (int(100*retreat_prob),122), (0, 255, 0), -1)



                  #calculate reward average over the last 1 and 5 seconds                  
                  #reset every 1 sec
                  #if last_reset_1 + 1 < time.time():  
                  #    reward_1 = 1-(hp_change_1/100) + attack_number_1/tick_count_1
                  #    hp_change_1 = 0
                  #    attack_number_1 = 0
                  #    print("reward_1: " + str(reward_1))
                  #    tick_count_1 = 0
                  #    last_reset_1 = time.time()
                  ##reset every 5 sec
                  #if last_reset_5 + 5 < time.time():  
                  reward_5 = (-20*hp_change_5) + attack_number_5 + (state.game_time/100)#/tick_count_5
                  cv2.rectangle(image_np,(51,122), (51+int((reward_5*6)),152), (255, 255, 255), -1)
                  cv2.rectangle(image_np,(51,122), (51,152), (0, 0, 255), -1)
                  #print("reward: " + str(reward_5) + " hp_change_5: " + str(hp_change_5) + " attack_number_5: " + str(attack_number_5))
                  hp_change_5 = 0
                  attack_number_5 = 0
                  #print(" reward_5: " + str(reward_5))
                  if last_reset_5 + 5 < time.time():  
                      #policy gradient estimation
                      estimate_policy_gradient_FD()
                      #print(policy.theta)
                      #print(policy.policy_gradient.reshape(4))
                      policy.theta += policy.policy_gradient.reshape(4,1)*0.01
                      #care if any value is < 0
                      if policy.theta[2] < 0:
                          policy.theta[2] = 0.0001
                      tick_count_5 = 0
                      policy.delta_R = []
                      policy.delta_theta = []
                      last_reset_5 = time.time()

                  #===============FEEDBACK==================
                  delta_theta_i = perturbate_policy()
                  policy.delta_theta.append(delta_theta_i)
                  updateR_i(delta_theta_i)
                  updateR_ref(reward_5)

                  delta_R = policy.R_i - policy.R_ref
                  #append the current action to delta_R[i]
                  policy.delta_R.append(delta_R)
                  print(str(policy.theta[0][0])+";"+str(policy.theta[1][0])+";"+str(policy.theta[2][0])+";"+str(policy.theta[3][0])+";"+str(reward_5))
                  #todo: print the reward and r_ref and see why its just declining
                  #print("======================================================================")               
      # ==================OPEN CV visualization=================
      cv2.imshow('AI_View', image_np)
      if (cv2.waitKey(25) & 0xFF == ord('q')):
          cv2.destroyAllWindows()
          break
      if gameover == True:
          print('gameover')
          cv2.destroyAllWindows()
          break;


# In[ ]:


#plot test for plotting thetas continuously for learning monitoring

#import matplotlib.pylab as pylab
#%matplotlib notebook
#def update_line(hl,x,p):
#    hl.set_xdata(x)
#    hl.set_ydata(p)
#    plt.draw()
#    plt.flush_events()
#x = np.arange(-6,6,0.1)
#playerHP_plot = 1
#thet = 1##
#
##fig=plt.figure()
#plt.show()
#for i in range(1,10):
#    p2 = i*playerHP_plot*(1/(1+np.exp(-2*(x+1)))-1/(1+np.exp(-10*(x-3))))
#    plt.plot(x,p2)
#    plt.draw()
 ##   fig.canvas.flush_events()
 #   time.sleep(1)
    #plt.gcf().clear()
    
    
    

#while True:
#    p = np.exp(x)#policy.theta[1]*playerHP_plot*(1/(1+np.exp(-2*(x+1)))-1/(1+np.exp(-10*(x-3))))
#    t, = pylab.plot(x,p)
#    time.sleep(1)
#    thet = 1
#    plt.show()
#p = policy.theta[1]*playerHP*(1/( 1+exp( -2*(x+1)) ) - 1/( 1+exp( -10*(x-3)) ) )

