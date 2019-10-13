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

# setup

    # loading in network and data

        #trainingData = data.TrainingData('D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMsTest.pgn')
        #print('data loaded')
        #train, test, val = torch.utils.data.random_split(trainingData, [165666, 9203, 9203])
        #print('data segregated')


    # Creates new network
#with open('D:\Machine Learning\DeepLearningChessAI\cudaTest.cnn', 'wb') as file:
#    pickle.dump(CNN.CNN().cuda(), file)

with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee2.cnn', 'rb') as file:
    network = pickle.load(file)
    network.cuda()

with open('D:\Machine Learning\DeepLearningChessAI\small_train_set.db', 'rb') as file:
    train_set = pickle.load(file).cudaDataset
    

with open('D:\Machine Learning\DeepLearningChessAI\small_test_set.db', 'rb') as file:
    test_set = pickle.load(file).cudaDataset

with open('D:\Machine Learning\DeepLearningChessAI\small_val_set.db', 'rb') as file:
    val_set = pickle.load(file).cudaDataset


    # initialize hyperparameters
batchSize = 1000
learningRate = 0.0001
epoch = 1

    # init optimizer
optimizer = optim.Adam(network.parameters(), learningRate)

# organize data

    # train data
train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
train_losses = list()

    # setting up test data
test_loader = torch.utils.data.DataLoader(test_set, 9203)
test_boards, test_results = next(iter(test_loader))
test_results = test_results.float().reshape([-1, 1]).cuda()
test_losses = list()

    # setting up validation data
val_loader = torch.utils.data.DataLoader(val_set, 9203)
val_boards, val_results = next(iter(val_loader))
val_results = val_results.float().reshape([-1, 1]).cuda()
val_losses = list()


# training loop

for epoch in range(epoch):

    for batch in train_loader:

        boards, results = batch

        print(boards)
        print(results)

        # converting type & reshaping
        results = results.float().reshape([-1, 1]).cuda()

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



# do i have protection against exploding/vanishing gradients/weights?
# cuda -> use GPU, its getting expensive
# why does output of ReLU exceed 1.0?
# why do weights "reset" after each epoch?