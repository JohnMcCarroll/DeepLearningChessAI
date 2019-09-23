import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import TrainingData as data
import pickle

# Data Prep

with open('D:\Machine Learning\DeepLearningChessAI\DatasetTest.db', 'rb') as file:
    train_set = pickle.load(file)

train_loader = torch.utils.data.DataLoader(train_set)

boards, results = next(iter(train_loader))