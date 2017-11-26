# LeagueAI
Imlementation of an A.I. Player for the videogame League of Legends based on Image Recognition, Discretization and Monte Carlo Policy Evaluation

LeagueAI View:
![Game View](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/002_LeagueAI.JPG)
![Game View](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/001_LeagueAI.jpg)

Game View:
![Game View](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/002_game.JPG)
![Game View](https://github.com/Oleffa/LeagueAI/blob/master/Pictures/001_game.jpg)

# Abstract
The task is to create an agent that is able to play 3rd person massive multiplayer online battle arena games (MOBA)
like League of Legends, Dota 2 and Heroes of the Storm with the same input as a human player. Image recognition is used to detect objects in the game environment. The environment is then discretized by dividing the screen content in a grid. The grid contains information for the agent like rewards and obstacles. With these informations an implementation of the Monte-Carlo Policy Evaluation is used to determine which area of the screen has to be clicked or how the agent has to position in order to get a maximum reward that leads to winning the game.

# History
## TODO
- Implement Decision making algorithm
- Make new demo video and update demo pictures
- Update report with newest progress
- Improve performance (or buy new computer $$$)
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


