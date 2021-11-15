#Deep Learning Chess Player

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

Retrieve Data from Chess.com:
Curate Data:
Train Neural Network:
Play a game:

- okay need to figure out data alteration step
- get full training loop
- get play loop running










Run training with a plastic ResNet:
```buildoutcfg
$ python research/shuffle/Gymnasium.py
```

### Results

After a few iterations on the idea, which are encapsulated in the "path", "sequential", and "shuffle" directories
and several experiment assays, it became clear that training time and performance were not significantly improved
compared to the state of the art alternative, ResNet, of similar depth. All tests involved training and inference on the CIFAR-10.
Although I feel there might be more work to do to exhaust this topic, it is my hypothesis that perhaps when operating in such a high
dimensional space as the paths and parameters of a ResNet, there is no need to manually manage connections
between blocks. When there's always room for further improvement (always a direction down the gradient), given high 
dimensionality, one can reach similar performance by training longer as opposed to manually pruning and adding new paths.
Essentially, the scale of the network mitigated the benefits of neuroplastic architecture search.
