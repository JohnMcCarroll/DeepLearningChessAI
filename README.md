# Deep Learning Chess Player

I applied the modern tool of Deep Learning to the classic game of Chess.

I was originally inspired to pursue Software Engineering by the amazing achievement of DeepMind
in creating Alpha Zero, which mastered the three great strategy games, Chess, Go, and Shogi, using the
same innovative reinforcement learning algorithm. As a lifelong Chess player, I was amazed at the intelligence
of a system capable of outperforming the world's best, hand-crafted chess AI (Stockfish) after just 8 hours of self play.
As a lifelong Chess enthusiast, it had been a goal of mine to create a Deep Learning Chess Player, ever
since I started learning to code. This project stands as a fond milestone in my 
development.

### Setup

Clone the repo to your local machine:
```buildoutcfg
$ git clone https://github.com/JohnMcCarroll/DeepLearningChessAI.git
```
Set up a virtual environment:
```buildoutcfg
$ python -m venv .
```

Activate your new virtual environment. This command is platform dependent, but for Linux it reads:
```buildoutcfg
$ source bin/activate 
```

Install dependencies:
```buildoutcfg
$ pip install -r requirements.txt
```

Retrieve Data from Chess.com (this may take a while):
```buildoutcfg
$ python src/data/DataRetrieval.py
```

Curate Data:
```buildoutcfg
$ python src/data/HashtableTraining.py
```

Train a Neural Network:
```buildoutcfg
$ python src/training/TrainingLoop.py
```

Begin a game against an AI Opponent:
```buildoutcfg
$ python src/playing/Player.py
```

The AI will begin with the white pieces and make its first move. It selects its move from the results of
a minimax tree search of the game space (the set of all possible future moves). The tree search
is completed to a fixed depth and breadth, which can be adjusted within the Player.py script.
The value of each board state used in the search is generated via inference from the 
trained CNN. The values are an approximate probability of White's chance of winning, based on the frequency
observed in our Chess.com dataset.


