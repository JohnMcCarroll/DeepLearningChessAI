import torch
import pickle
import statistics
import gc
import json

### A script used for quick data curation and manipulation (ie. calculating probabilities, removing samples, etc.)

# assumes data is list and board is in string representation
def probability(data):
    # get dataset and set up dictionary
    table = dict()
    #counter = 0
    # convert list into dictionary where key is input and value is list of game results
    for datum in data:

        # pair boardstates with list of results
        if datum[0] in table:
            table[datum[0]] = table[datum[0]] + [datum[1]]
        else:
            table[datum[0]] = [datum[1]]

        #counter += 1
        #print(counter)

    print("list converted to dictionary")

    with open(r'D:\Machine Learning\DeepLearningChessAI\Data\StringKeyProb_table.db', 'wb') as file:
        pickle.dump(table, file)

    print("comparison of table to list:")
    print("list length:")
    print(len(data))
    print("hashtable keys:")
    print(len(table.keys()))

    sum = 0

    # calculate probabilities
    for position in table.keys():
        sum += len(table[position])
        table[position] = statistics.mean(table[position])

    print("hashtable elements:")
    print(sum)

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

def boardToString(board):
    string = ""
    for i in range(0,64):
        spaceVector = board[0:12, int(i / 8), i % 8]
        channel = (spaceVector == 1).nonzero()       #get the piece type that resides on square

        try:
            if channel.size()[0] == 0:                  # if empty
                string += "E,"
            elif channel.item() == 0:
                string += "WK,"
            elif channel.item() == 1:
                string += "WQ,"
            elif channel.item() == 2:
                string += "WR,"
            elif channel.item() == 3:
                string += "WB,"
            elif channel.item() == 4:
                string += "WN,"
            elif channel.item() == 5:
                string += "WP,"
            elif channel.item() == 6:
                string += "BK,"
            elif channel.item() == 7:
                string += "BQ,"
            elif channel.item() == 8:
                string += "BR,"
            elif channel.item() == 9:
                string += "BB,"
            elif channel.item() == 10:
                string += "BN,"
            elif channel.item() == 11:
                string += "BP,"

        except:
            print(board)
            print(channel)
            return None

    # whose turn?
    turn = board[13, 0, 0]
    if turn == 0:
        string += "W"
    else:
        string += "B"

    return string

def averageResults(data, index):

    size = len(data.keys())
    start = index * int(size / 32)
    end = (index + 1) * int(size / 32)

    print("start:")
    print(start)
    print("end:")
    print(end)

    keys = list(data.keys())
    newDataset = list()

    # calculate probabilities
    for index in keys:                                       # range(start, end):     ***altered to do whole table
        # convert to tensor rep of board
        board = stringToBoard(index)
        # add new pair to new dataset
        newDataset.append((board, statistics.mean(data[index]))) 
        # remove old pair from dataset
        del data[index]
        # garbage collect
        gc.collect()

    print("altered data")
    return newDataset  

def stringToBoard(stringBoard):
    # initialize board state tensor
    board = torch.zeros([14, 8, 8])

    # set turn
    if stringBoard[-1] == "B":
        board[12:14, :, :] = 1

    # place pieces
    boardFields = stringBoard.split(",")
    for index in range(0,64):
        col = index % 8
        row = int(index / 8)

        if boardFields[index] == "E":
            continue
        elif boardFields[index] == "WK":
            board[0, row, col] = 1
        elif boardFields[index] == "WQ":
            board[1, row, col] = 1
        elif boardFields[index] == "WR":
            board[2, row, col] = 1
        elif boardFields[index] == "WB":
            board[3, row, col] = 1
        elif boardFields[index] == "WN":
            board[4, row, col] = 1
        elif boardFields[index] == "WP":
            board[5, row, col] = 1
        elif boardFields[index] == "BK":
            board[6, row, col] = 1
        elif boardFields[index] == "BQ":
            board[7, row, col] = 1
        elif boardFields[index] == "BR":
            board[8, row, col] = 1
        elif boardFields[index] == "BB":
            board[9, row, col] = 1
        elif boardFields[index] == "BN":
            board[10, row, col] = 1
        elif boardFields[index] == "BP":
            board[11, row, col] = 1
    
    return board

# data duplication of non-unique boardstates:
"""
originalDatasetPath = r'D:\Machine Learning\DeepLearningChessAI\Data\hashtableDatasetA.txt'
newDatasetPath = r'D:\Machine Learning\DeepLearningChessAI\Data\QualityDataset.txt'
with open(newDatasetPath, 'a') as newFile:
    with open(originalDatasetPath, 'r') as OGfile:
        for line in OGfile:
            fields = line.split(" ~ ")
            value = float(fields[1])
            if value != 0.0 and value != 1.0 and value != 0.5:
                newFile.write(line)
"""



# data combinations

"""
import os
fullData = list()
index = 0

for filename in os.listdir(r"D:\ChessEngine\DeepLearningChessAI\Data\prob_data"):
    with open(r"D:\ChessEngine\DeepLearningChessAI\Data\prob_data\\" + filename, 'rb') as file:
        data = pickle.load(file)

    fullData.extend(data)

    print(filename + " added")

    data = []

    gc.collect()

    with open(r'D:\ChessEngine\DeepLearningChessAI\Data\prob_data\prob_data' + str(index) + '.db', 'wb') as file:
        pickle.dump(fullData, file)
        index += 1
"""

### Next Steps:
# compile hashtable data into list of (tensor, double) tuples
# remove positions that only appeared once (eliminate binary bias)
# add next move probabilities to data & CNN architecture
