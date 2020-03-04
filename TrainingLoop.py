import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import DataAlteration as Data
import numpy as np
import pickle
import CNN
import matplotlib.pyplot as plt
import time
import gc
import os
import linecache

# define functions
def parseLine(line, dataset):
    fields = line.split(" ~ ")
    tensorBoard = Data.stringToBoard(fields[0])
    dataset.append((tensorBoard, float(fields[1])))

# setup
    # initialize hyperparameters & variables
batchSize = 50
learningRate = 0.0001
epoch = 1
subepoch = 10
test_set_size = 10000
datasetFilepath = r'D:\Machine Learning\DeepLearningChessAI\Data\QualityDataset.txt'
dataset = list()
test_set = list()

    # loading in network and data
    # Creates new network
network = CNN.CNN().cuda()

    # partition a test set
datasetSize = 0
with open(datasetFilepath, 'r') as file:
    for i, line in enumerate(file):
        # add datum to test_set
        if i < test_set_size:
            parseLine(line, test_set)
        datasetSize = i

    # establish length of subepochs
subepochSize = int((datasetSize + 1) / subepoch)

    # init optimizer
optimizer = optim.Adam(network.parameters(), learningRate)

# organize data

    # train data
# train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
train_losses = list()

    # setting up test data
test_loader = torch.utils.data.DataLoader(test_set, test_set_size)
test_boards, test_results = next(iter(test_loader))
test_results = test_results.float().reshape([-1, 1]).cuda()     #switch to gpu
test_boards = test_boards.cuda()
test_losses = list()

    # setting up validation data
# val_loader = torch.utils.data.DataLoader(val_set, 9203)
# val_boards, val_results = next(iter(val_loader))
# val_results = val_results.float().reshape([-1, 1]).cuda()       #switch to gpu
# val_losses = list()


# training loop

#add loops for testing hyperparams / architectures

for epoch in range(epoch):

    for subepoch in range(subepoch):
        # set up partition of dataset as train_set
        train_set = list()
        with open(datasetFilepath, 'r') as file:
            for i, line in enumerate(file):
                # add board position to train_set
                if test_set_size + subepochSize*subepoch <= i:
                    parseLine(line, train_set)

                # break from loop once out of scope
                if i >= subepochSize*(subepoch + 1) + test_set_size:
                    break

        # create data_loader from train_set
        train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)

        # train on subepoch
        for batch in train_loader:

            boards, results = batch

            # converting type & reshaping
            results = results.float().reshape([-1, 1]).cuda()       #switch to gpu
            boards = boards.cuda()

            # calculating loss
            preds = network(boards)
            loss = F.mse_loss(preds, results)

            train_losses.append(loss.item())    # store train loss for batch

            # calculating gradients
            optimizer.zero_grad()   #clear out accumulated gradients
            loss.backward()
            optimizer.step() # updating weights

            # benchmark if learning
            test_preds = network(test_boards)
            test_loss = F.mse_loss(test_preds, test_results)
            test_losses.append(test_loss.item())

plt.plot(test_losses)
plt.ylabel('test loss')
plt.xlabel('batch number')
plt.show()

plt.plot(train_losses)
plt.ylabel('train loss')
plt.xlabel('prediction number')
plt.show()

# free up RAM
data = []
test_set = []
gc.collect()

# save network
torch.save(network, r'D:\Machine Learning\DeepLearningChessAI\Networks\QualityControl.cnn')