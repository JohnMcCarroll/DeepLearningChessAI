import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import src.data.DataAlteration as Data
import numpy as np
import pickle
import src.playing.CNN as CNN
import matplotlib.pyplot as plt
import time
import gc
import os
import linecache

### The TrainingLoop script loads or initializes a CNN and trains it for a specified number of epochs on a specified dataset

# define functions
def parseLine(line, dataset):
    fields = line.split(" ~ ")
    tensorBoard = Data.stringToBoard(fields[0])
    dataset.append((tensorBoard, float(fields[1])))


if __name__ == '__main__':
    # setup
        # initialize hyperparameters & variables
    batchSize = 50
    learningRate = 0.0001
    epoch_count = 1
    subepoch_count = 100
    test_set_size = 10000
    datasetFilepath = r'games.txt'
    dataset = list()
    test_set = list()

        # loading in network and data
        # Creates new network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = CNN.CNN().to(device)

        # partition a test set
    datasetSize = 0
    with open(datasetFilepath, 'r') as file:
        for i, line in enumerate(file):
            # add datum to test_set
            if i < test_set_size:
                parseLine(line, test_set)
            datasetSize = i

        # establish length of subepochs
    subepochSize = int((datasetSize + 1) / subepoch_count)

        # init optimizer
    optimizer = optim.Adam(network.parameters(), learningRate)

    # organize data

        # train data
    # train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
    train_losses = list()

        # setting up test data
    test_loader = torch.utils.data.DataLoader(test_set, test_set_size)
    test_boards, test_results = next(iter(test_loader))
    test_results = test_results.float().reshape([-1, 1]).to(device)     #switch to gpu
    test_boards = test_boards.to(device)
    test_losses = list()

        # setting up validation data
    # val_loader = torch.utils.data.DataLoader(val_set, 9203)
    # val_boards, val_results = next(iter(val_loader))
    # val_results = val_results.float().reshape([-1, 1]).cuda()       #switch to gpu
    # val_losses = list()


    # training loop

    #add loops for testing hyperparams / architectures

    for current_epoch in range(epoch_count):

        with open(datasetFilepath, 'r') as file:
            # Skip the test set portion at the beginning of the file
            for _ in range(test_set_size):
                next(file, None)
                
            for current_subepoch in range(subepoch_count):
                # set up partition of dataset as train_set
                train_set = list()
                for _ in range(subepochSize):
                    line = next(file, None)
                    if line is not None:
                        parseLine(line, train_set)
                    else:
                        break

                # create data_loader from train_set
                if train_set:
                    train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)

                # train on subepoch
                subepoch_loss_sum = 0.0
                batch_count = 0
                for batch in train_loader:

                    boards, results = batch

                    # converting type & reshaping
                    results = results.float().reshape([-1, 1]).to(device)       #switch to gpu
                    boards = boards.to(device)

                    # calculating loss
                    preds = network(boards)
                    loss = F.mse_loss(preds, results)

                    train_losses.append(loss.item())    # store train loss for batch
                    subepoch_loss_sum += loss.item()
                    batch_count += 1

                    # calculating gradients
                    optimizer.zero_grad()   #clear out accumulated gradients
                    loss.backward()
                    optimizer.step() # updating weights

                # benchmark if learning at the end of the subepoch
                with torch.no_grad():
                    test_preds = network(test_boards)
                    test_loss = F.mse_loss(test_preds, test_results)
                    test_losses.append(test_loss.item())
                
                # log progress to the console
                avg_train_loss = subepoch_loss_sum / batch_count if batch_count > 0 else 0
                print(f"Epoch [{current_epoch+1}/{epoch_count}] Subepoch [{current_subepoch+1}/{subepoch_count}] - Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss.item():.4f}")

    plt.plot(test_losses)
    plt.ylabel('test loss')
    plt.xlabel('subepoch number')
    plt.savefig('test_loss.png')
    plt.clf()

    plt.plot(train_losses)
    plt.ylabel('train loss')
    plt.xlabel('prediction number')
    plt.savefig('train_loss.png')
    plt.clf()

    # free up RAM
    data = []
    test_set = []
    gc.collect()

    # save network
    torch.save(network, r'BetaZero.cnn')
