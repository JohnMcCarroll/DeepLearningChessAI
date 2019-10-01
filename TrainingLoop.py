import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import TrainingData as data
import pickle
import CNN
import matplotlib.pyplot as plt

# loading in 

# Creates new network
#with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee1.cnn', 'wb') as file:
#    pickle.dump(CNN.CNN(), file)

with open('D:\Machine Learning\DeepLearningChessAI\DatasetTest.db', 'rb') as file:
    train_set = pickle.load(file)

with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee1.cnn', 'rb') as file:
    network = pickle.load(file)

# initialize hyperparameters
batchSize = 100
learningRate = 0.001
epoch = 2

# setting up training data
train_set, validation_set, dummy_set = torch.utils.data.random_split(train_set, [166000, 18000, 72])        #prob shouldnt be random split in val is benchmark between networks
train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
optimizer = optim.Adam(network.parameters(), learningRate)

# setting up validation data
validation_loader = torch.utils.data.DataLoader(validation_set, 1000)       # also shouldnt be random...
val_boards, val_results = next(iter(validation_loader))
val_results = val_results.float().reshape([-1, 1])
val_losses = list()

for epoch in range(epoch):
    for batch in train_loader:
        boards, results = batch

        # converting type & reshaping
        results = results.float().reshape([-1, 1])

        # calculating loss
        preds = network(boards)
        loss = F.mse_loss(preds, results)

        # calculating gradients
        optimizer.zero_grad()   #clear out accumulated gradients
        loss.backward()
        optimizer.step() # updating weights

        # benchmark if learning
        preds = network(val_boards)
        loss = F.mse_loss(preds, val_results)
        val_losses.append(loss.item())

    plt.plot(val_losses)
    plt.ylabel('validation loss')
    plt.xlabel('batch number')
    plt.show()

# save network
with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee1.cnn', 'wb') as file:
    pickle.dump(network, file)



# do i have protection against exploding/vanishing gradients/weights?