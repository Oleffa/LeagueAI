# LeagueAI
Implementation of an A.I. Player for the videogame League of Legends based on Image Recognition using TensorFlow, OpenCV, Discretization and Gradient Optimization. 

Demo video: https://www.youtube.com/watch?v=KRWFCaXfOTk

# Abstract
The task is to create an agent that is able to play 3rd person massive multiplayer online battle arena games (MOBA)
like League of Legends, Dota 2 and Heroes of the Storm with the same input as a human player. Image recognition is used to detect objects in the game environment. Currently the TensorFlow object detection is able to detect the player Character (Vayne), enemy minions and enemy towers. The green grid is a representation of the discretized game environment with each square representing a state. States that contain objects are colored differently. For example a state that contains a minion is marked with a blue rectangle and a tower with a white one. Sample between Game view and the AI's view:
![AI_002](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/002_LeagueAI.JPG)
![Game_002](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/002_game.jpg)

Tower Recognition:
![AI_001](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/001_LeagueAI.JPG)
![Game_001](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/001_game.jpg)

These informations allow us to calculate an action like attack a certain state, run away from a threat in a state like a tower or approach enemies. This is done using gradient optimization on a custom policy. Based on the loss or gain of the players HP, the number of attacks executed and the time we survived so far we then calculate a reward which is fed back to the system to improve the decision making while playing.

# History
## TODO
- Teach more objects to the model
- Add more interactions between game objects (make decisions based on more factors than just distance to each other and own hp)
- Improve performance (or buy new computer $$$)
- React to getting stuck by recalling and moving back to lane -> Allows the bot to play games completely on its own

## 2.12.2017-4.12.2017
Finalizing the system to a state that can be presented.
- Implemented policy finding
- Improved reward calculation and learning
- Added mode to move autonomously to lane and wait for minions
- Tried to implement abilites and a recall function but failed due to problems with win32api and keystroke/mouse actions sending to the HUD (apparently it has something to do with the drivers and that python generated key events are different than events created by actual hardware)


## 26.11.2017
Implemented a function to calculate rewards based on player HP, number of attacks executed in a certain time frame as well as the time the agent survived so far.
Implemented functions to determine a threat level from certain game objects. This info will be used by the gradient optimization in the future.

## 22.11.2017
Planned how to implement the decision making. It will be based on a few parameters and also factor in the distance to the closest enemy tower into its decision making. Also it is required to know how much hp the player character has as a measure of "reward" for an action.
Therefore a function to determine the percent of HP the player character has was implemented counting the green/non-green pixels of the health bar in the HUD.
## 20.11.2017
Added a hardcoded logic to make decisions (attack, reposition, approach enemy, and move to enemy base) based on the distance to the closest enemy. If an enemy is too close reposition and if its to far approach it. If it is in attakc range and not to close attack the enemy.
The results were quite good and the agent is keeping a safe distance to enemies and attacks them when possible. The additional calculations drain my notebooks resources and the calculation algorithm just performs every 1.5 seconds which is barely enough to react fast to changes in the game. Better hardware and using GPU acceleration would be necessary.
## 14.11.2017
Wrote new helper functions to also generate a precise x|y position of minions. These positions are then used to find out in which state of the discretized enviornment they are. Next step implement decision making based on the state matrix, generating a matrix containing the rewards and implement the Markovian Process for determining the best action to take
## 13.11.2017
Testing of an improved model which can now detect the player champion, enemy minions and enemy towers. The resulting accuracy was surprisingly high. The champion and towers are detected with accuracies of over 90% and minions are even detected when overlaping with other minions. I guess it was worth it to spend 2 days recording and labeling 600 pictures as a training set...
## 4.11.2017
Taking more pictures of the player character in more situations to improve detection performance. Also starting to train the model to recognize enemy minions. Took about 350 images of the player character and 200 of enemy minions.
## 26.10.2017
https://youtu.be/iJSQLHRssiI
So far the bot can detect itself in the game. Around its position a grid is established with each cell representing a state in which the player could be. So far the selection of the next state to move to is hard coded. The player just moves to the top right state all the time. This results in the bot moving to the top right corner of the map where the enemy nexus is.


