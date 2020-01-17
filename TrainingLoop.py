import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
# import TrainingData as data
import pickle
import CNN
import matplotlib.pyplot as plt
import time
import gc
import os


# setup

    # loading in network and data

    # Creates new network
network = CNN.CNN().cuda()

# with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee2.cnn', 'rb') as file: 
#     network = pickle.load(file)
#     network.cuda()

# with open(r'D:\Machine Learning\DeepLearningChessAI\Data\ratioDataset.db', 'rb') as file:
#     train_set = pickle.load(file)


# test_set = train_set[0:1000]
# val_set = train_set[1000:2000]
# train_set = train_set[2001:len(train_set)]


# with open(r'D:\Machine Learning\DeepLearningChessAI\val_set2.list', 'rb') as file:
#     test_set = pickle.load(file)

# with open(r'D:\Machine Learning\DeepLearningChessAI\train_set2.list', 'rb') as file:
#     val_set = pickle.load(file)

with open(r"D:\ChessEngine\DeepLearningChessAI\Data\prob_data\prob_dataset31.db", 'rb') as file:
    test_set = pickle.load(file)


    # initialize hyperparameters
batchSize = 100
learningRate = 0.0001
epoch = 3

    # init optimizer
optimizer = optim.Adam(network.parameters(), learningRate)

# organize data

    # train data
# train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
train_losses = list()

    # setting up test data
test_loader = torch.utils.data.DataLoader(test_set, 1000)
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

    # for each file of dataset
    for filename in os.listdir(r"D:\ChessEngine\DeepLearningChessAI\Data\prob_data"):
        data = []
        gc.collect()

        # load in data
        with open(r"D:\ChessEngine\DeepLearningChessAI\Data\prob_data\\" + filename, 'rb') as file:
            data = pickle.load(file)

        # organize data
        print(filename)

        if filename == "prob_dataset31.db":        # skip if test set
            continue
        else:
            train_loader = torch.utils.data.DataLoader(data, batchSize, shuffle=True)

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
# with open(r'D:\ChessEngine\DeepLearningChessAI\Networks\Skipper5a.cnn', 'wb') as file:
#    pickle.dump(network, file)

torch.save(network, r'D:\ChessEngine\DeepLearningChessAI\Networks\Skipper5a.cnn')

