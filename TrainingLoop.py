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

import time

# setup

    # loading in network and data

"""
trainingData = data.TrainingData('D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMs.pgn')
print('data loaded')
print(len(trainingData.dataset))
"""
with open(r"D:\Machine Learning\DeepLearningChessAI\full_dataset.db", 'rb') as file:
    #pickle.dump(trainingData.dataset, file, protocol=2)
    trainingData = pickle.load(file)

print('dataset loaded')

train, test, val = torch.utils.data.random_split(trainingData, [22525324, 1251407, 1251407])
print('data segregated')

train_set = list()
test_set = list()
val_set = list()

for index in train.indices:
    train_set.append(train.dataset[index])

for index in test.indices:
    test_set.append(test.dataset[index])

for index in val.indices:
    val_set.append(val.dataset[index])

    # Creates new network
#with open('D:\Machine Learning\DeepLearningChessAI\cudaTest.cnn', 'wb') as file:
#    pickle.dump(CNN.CNN().cuda(), file)

#with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee2.cnn', 'rb') as file: 
#    network = pickle.load(file)
#    network.cuda()

with open(r'D:\Machine Learning\DeepLearningChessAI\train_set.db', 'wb') as file:
    #train_set = pickle.load(file)
    #print(len(train_set))
    pickle.dump(train_set, file, protocol=2)

with open(r'D:\Machine Learning\DeepLearningChessAI\test_set.db', 'wb') as file:
    #test_set = pickle.load(file)
    pickle.dump(test_set, file, protocol=2)

with open(r'D:\Machine Learning\DeepLearningChessAI\val_set.db', 'wb') as file:
    #val_set = pickle.load(file)
    pickle.dump(val_set, file, protocol=2)


    # initialize hyperparameters
batchSize = 1000
learningRate = 0.0001
epoch = 1

    # init optimizer
#optimizer = optim.Adam(network.parameters(), learningRate)

# organize data

    # train data
print(len(train_set))
print(len(test_set))
print(len(val_set))

time.sleep(10)




train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
train_losses = list()

    # setting up test data
test_loader = torch.utils.data.DataLoader(test_set, 9203)
test_boards, test_results = next(iter(test_loader))                 #whole test set?***
test_results = test_results.float().reshape([-1, 1]).cuda()     #switch to gpu
test_losses = list()

    # setting up validation data
val_loader = torch.utils.data.DataLoader(val_set, 9203)
val_boards, val_results = next(iter(val_loader))
val_results = val_results.float().reshape([-1, 1]).cuda()       #switch to gpu
val_losses = list()


# training loop

for epoch in range(epoch):

    for batch in train_loader:

        boards, results = batch

        print(boards.size())
        print(results.size())

        # converting type & reshaping
        results = results.float().reshape([-1, 1]).cuda()       #switch to gpu

        # calculating loss
        preds = network(boards)
        loss = F.mse_loss(preds, results)

        train_losses.append(loss.item())    # store train loss for batch

        # calculating gradients
        optimizer.zero_grad()   #clear out accumulated gradients
        loss.backward()
        optimizer.step() # updating weights

        # benchmark if learning
        preds = network(test_boards)
        loss = F.mse_loss(preds, test_results)
        test_losses.append(loss.item())

    plt.plot(test_losses)
    plt.ylabel('test loss')
    plt.xlabel('batch number')
    plt.show()

# save network
#with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee2.cnn', 'wb') as file:
 #   pickle.dump(network, file)




# why do weights "reset" after each epoch?