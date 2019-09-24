import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import TrainingData as data
import pickle
import CNN

# Data Prep

with open('D:\Machine Learning\DeepLearningChessAI\DatasetTest.db', 'wb') as file:
    train_set = data.TrainingData('D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMsTest.pgn')
    pickle.dump(train_set, file)

with open('D:\Machine Learning\DeepLearningChessAI\DatasetTest.db', 'rb') as file:
    train_set = pickle.load(file)

train_loader = torch.utils.data.DataLoader(train_set, 50)

boards, results = next(iter(train_loader))

network = CNN.CNN()
pred = network(boards)

print(pred)
print(results)

# figure out why network output so uniform? save / load network? shuffle data before training loop? and implement training loop?