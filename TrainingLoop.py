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


# setup

    # loading in network and data

    # Creates new network
with open('D:\\Machine Learning\\DeepLearningChessAI\\Networks\\Skipper5.cnn', 'wb') as file:
    network = CNN.CNN().cuda()
    pickle.dump(network, file)

# with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee2.cnn', 'rb') as file: 
#     network = pickle.load(file)
#     network.cuda()

with open(r'D:\Machine Learning\DeepLearningChessAI\Data\ratioDataset.db', 'rb') as file:
    train_set = pickle.load(file)


test_set = train_set[0:1000]
val_set = train_set[1000:2000]
train_set = train_set[2001:len(train_set)]


# with open(r'D:\Machine Learning\DeepLearningChessAI\val_set2.list', 'rb') as file:
#     test_set = pickle.load(file)

# with open(r'D:\Machine Learning\DeepLearningChessAI\train_set2.list', 'rb') as file:
#     val_set = pickle.load(file)


    # initialize hyperparameters
batchSize = 1000
learningRate = 0.0001
epoch = 3

    # init optimizer
optimizer = optim.Adam(network.parameters(), learningRate)

# organize data

    # train data
train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
train_losses = list()

    # setting up test data
test_loader = torch.utils.data.DataLoader(test_set, 9203)
test_boards, test_results = next(iter(test_loader))                 #whole test set?***
test_results = test_results.float().reshape([-1, 1]).cuda()     #switch to gpu
test_boards = test_boards.cuda()
test_losses = list()

    # setting up validation data
val_loader = torch.utils.data.DataLoader(val_set, 9203)
val_boards, val_results = next(iter(val_loader))
val_results = val_results.float().reshape([-1, 1]).cuda()       #switch to gpu
val_losses = list()


# training loop

#add loops for testing hyperparams / architectures

for epoch in range(epoch):

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



# save network
with open(r'D:\Machine Learning\DeepLearningChessAI\Networks\Skipper5.cnn', 'wb') as file:
   pickle.dump(network, file)

