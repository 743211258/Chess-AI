# Overview
Chess-AI is my first individual project. It is similar to Stockfish, or LCZero, that predicts the winning rate and next best move based on a given position.
My implementation mostly refers to the famous paper "Mastering the game of Go without human knowledge", using a CNN and Resnet to extract features out of the inputs,
then send these information to both policy head and value head for computation. I also implemented a Monte-Carlo Search algorithm to supplement the neural network for a better result.

The Neural network is trained on self-play methods, a type of deep reinforcement learning. Two players are generated and they plays a move alternatively until the game is end or draw. 
The move is chosen based on both Monte-Carlo Search and a series of parameters that ensures exploration. The result is saved in a form that can be used for training.

# Functions
The following list is a brief description of each function that I had uploaded.

|Functions                             |Descriptions                                                                                                        |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|archive_data.jsonl                    |A file to store past training data.(Over 30000 lines)                                                               |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|cnn.py                                |The implementation of the neural network.                                                                           |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|data.py                               |Based on mcts.py. This file enables self play and record outcomes into training_data.json.                          |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|game.py                               |This file calculated all possible legal moves in uci notation. Each move is stored in a list called move_id.        |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|input.py                              |Read a pgn file or convert a chess board to tensors.                                                                |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|lichess_db_standard_rated_2013-01.pgn |pgn file downloaded from lichess open database.                                                                     |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|load_data.py                          |Load a line of data, or data in batches to train the neural network.                                                |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|  
|main.py                               |The main training process happens here.                                                                             |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|mcts.py                               |The implementation of Monte Carlo Search.                                                                           |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|model_checkpoint.pth                  |Weights of the neural network after some training.                                                                  |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|train.py                              |Train the neural network in batches.                                                                                |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|training_data.json                    |A file to store current training data.                                                                              |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
