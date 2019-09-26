import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import TrainingData as data
import pickle
import CNN

# Data Prep

#with open('D:\Machine Learning\DeepLearningChessAI\DatasetTest.db', 'wb') as file:
#    train_set = data.TrainingData('D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMsTest.pgn')
#    pickle.dump(train_set, file)

with open('D:\Machine Learning\DeepLearningChessAI\DatasetTest.db', 'rb') as file:
    train_set = pickle.load(file)

train_loader = torch.utils.data.DataLoader(train_set, 100)

boards, results = next(iter(train_loader))

# converting type... might have to redownload data

results = results.long()

network = CNN.CNN()

# calculating loss
preds = network(boards)
print(preds.type())
loss = F.cross_entropy(preds, results)
print('before:')
print(loss.item())

# calculating gradients
loss.backward()
optimizer = optim.Adam(network.parameters(), 0.01)
optimizer.step() # updating weights

# checking if learning
pred = network(boards)
loss = F.cross_entropy(preds, results)
print('after:')
print(loss.item())

# save / load network? shuffle data before training loop? and implement training loop?

# classification issue -> labels from 0-1, interpretted as two classes. my output layer = 1 node, interpretted as one class...
# training issue / tensor type issue -> ***preds.type() = float***  ...need network to make predictions as double / float data types, long is only in integers... might explain why loss is not decreasing / why network only outputs 0 lol