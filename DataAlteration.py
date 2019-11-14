import torch
import pickle
#import TrainingData as td
import Node
import statistics

# script to change data results from win/loss to win probability

# assumes data is list
def probability(data):
    # get dataset and set up dictionary
    table = dict()
    #counter = 0
    # convert list into dictionary where key is input and value is list of game results
    for datum in data:
        
        # convert tensor to immutable tuple
        t = datum[0].flatten()
        t = t.tolist()
        t = [round(x, 3) for x in t]
        boardTuple = tuple(t)

        # pair boardstates with list of results
        if boardTuple in table:
            table[boardTuple] = table[boardTuple] + [datum[1]]
        else:
            table[boardTuple] = [datum[1]]

        #counter += 1
        #print(counter)

    print("list converted to dictionary")
    
    # calculate probabilities
    for position in table.keys():
        table[position] = statistics.mean(table[position])

    # convert dict to list of tuples
    newDataset = [(k, v) for k, v in table.items()]

    # convert boardstate tuple to tensor
    for index in range(0, len(newDataset)):
        board = list(newDataset[index][0])
        board = torch.FloatTensor(board)
        board = board.reshape(14,8,8)
        newDataset[index] = (board, newDataset[index][1])


    print("altered data")
    return newDataset
    
            
def initialBoard():
        # initialize board state tensor
        board = torch.zeros([14, 8, 8])

        # White King
        board[0, 7, 4] = 1
        # White Queen
        board[1, 7, 3] = 1
        # White Rooks
        board[2, 7, 0] = 1
        board[2, 7, 7] = 1
        # White Bishops
        board[3, 7, 2] = 1
        board[3, 7, 5] = 1
        # White Knights
        board[4, 7, 1] = 1
        board[4, 7, 6] = 1
        # White Pawns
        board[5, 6, :] = 1
        # Black King
        board[6, 0, 4] = 1
        # Black Queen
        board[7, 0, 3] = 1
        # Black Rooks
        board[8, 0, 0] = 1
        board[8, 0, 7] = 1
        # Black Bishops
        board[9, 0, 2] = 1
        board[9, 0, 5] = 1
        # Black Knights
        board[10, 0, 1] = 1
        board[10, 0, 6] = 1
        # Black Pawns
        board[11, 1, :] = 1

        return board

### data recovery:
#data = td.TrainingData(r'D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMs.pgn')

with open(r'D:\Machine Learning\DeepLearningChessAI\Data\full_dataset.db', 'rb') as file:
    data = pickle.load(file)


for datum in data:
    print(datum[1])
    print(type(datum[1]))
    if datum[1] != 0.0 and datum[1] != 0.5 and datum[1] != 1.0:
        print(datum)
        input()

# data = probability(data)

# print(data[0])
# print(data[4])
# print(data[10])
# print(data[12])
# print(data[19])
# print(data[24])
# print(data[39])
# print(data[40])
# print(data[41])
# print(data[42])
# print(data[43])
# print(data[44])
# print(data[45])
# print(data[46])

# with open(r'D:\Machine Learning\DeepLearningChessAI\Data\full_dataset.db', 'wb') as file:
#     pickle.dump(data, file)





### data verification:

# with open(r'D:\Machine Learning\DeepLearningChessAI\full_dataset2.db', 'rb') as file:
#     data = pickle.load(file)

# with open(r'D:\Machine Learning\DeepLearningChessAI\test_set.db', 'rb') as file:
#     data2 = pickle.load(file)

# with open(r'D:\Machine Learning\DeepLearningChessAI\val_set.db', 'rb') as file:
#     data3 = pickle.load(file)

# full_data = data1 + data2 + data3

# with open(r'D:\Machine Learning\DeepLearningChessAI\full_dataset.db', 'wb') as file:
#     pickle.dump(full_data, file)


# print("size:")
# print(len(data))
# input()


# with open(r'D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMs.pgn', 'rb') as file:
#    data = pickle.load(file)           



# data = list()
# data.append((initialBoard(), 0.5))
# data.append((initialBoard(), 1.0))


# data = probability(data)

# print('data:')
# print(data)

