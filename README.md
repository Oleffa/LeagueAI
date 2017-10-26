# LeagueAI
Imlementation of an A.I. Player for the videogame League of Legends based on Image Recognition, Discretization and Monte Carlo Policy Evaluation

# Abstract
The task is to create an agent that is able to play 3rd person massive multiplayer online battle arena games (MOBA)
like League of Legends, Dota 2 and Heroes of the Storm with the same input as a human player. Image recognition is used to detect objects in the game environment. The environment is then discretized by dividing the screen content in a grid. The grid contains information for the agent like rewards and obstacles. With these informations an implementation of the Monte-Carlo Policy Evaluation is used to determine which area of the screen has to be clicked or how the agent has to position in order to get a maximum reward that leads to winning the game.

# Demo
##26.10.2017
https://www.youtube.com/edit?o=U&video_id=iJSQLHRssiI
So far the bot can detect itself in the game. Around its position a grid is established with each cell representing a state in which the player could be. So far the selection of the next state to move to is hard coded. The player just moves to the top right state all the time. This results in the bot moving to the top right corner of the map where the enemy nexus is.


