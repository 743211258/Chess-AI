# Overview
Chess-AI is my first individual project. It is similar to Stockfish, or LCZero, that predicts the winning rate and next best move based on a given position.
My implementation mostly refers to the famous paper "Mastering the game of Go without human knowledge", using a CNN and Resnet to extract features out of the inputs,
then send these information to both policy head and value head for computation. I also implemented a Monte-Carlo Search algorithm to supplement the neural network for a better result.

The Neural network is trained on self-play methods, a type of deep reinforcement learning. Two players are generated and they plays a move alternatively until the game is end or draw. 
The move is chosen based on both Monte-Carlo Search and a series of parameters that ensures exploration. The result is saved in a form that can be inputted into the neural network for training.

# Functions
The following list is a brief description of each function that I had uploaded.

|Functions                          |Descriptions                                                                                                        |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------|
