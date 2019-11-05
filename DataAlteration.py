import torch
import pickle
import TrainingData as td

# script to change data results from win/loss to win probability

# assumes data is list
def probability(data):
    # get dataset and set up dictionary
    table = dict()
    counter = 0
    # convert list into dictionary where key is input and value is list of game results
    for datum in data:
        keys = table.keys()
        # if the input position is already 
        newPos = True
        for key in keys:
            if torch.equal(key, datum[0]):
                table[key] = table[key] + [datum[1]]
                newPos = False
                break

        # if input position is new
        if newPos:
            table[datum[0]] = [datum[1]]

        counter += 1
        print(counter)

    print("list converted to dictionary")

    # calculate probabilities
    for position in table.keys():
        elements = table[position]
        num_elem = len(elements)
        sum_elem = sum(elements)
        table[position] = sum_elem / num_elem


    # convert dict to list of tuples
    newDataset = [(k, v) for k, v in table.items()]

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


with open(r'D:\Machine Learning\DeepLearningChessAI\small_test_set.db', 'rb') as file:      #dataset just list...
    data = pickle.load(file)           

data = probability(data)
print(data[0])
print(data[1])
print(data[2])
print(data[10])

with open(r'D:\Machine Learning\DeepLearningChessAI\newDatums.idfk', 'wb') as file:
    pickle.dump(data, file)