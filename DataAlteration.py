import torch
import pickle
#import TrainingData as td
import Node
import statistics

# script to change data results from win/loss to win probability

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

def displayBoard(board):
    # create display
    display = [['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '], ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
               ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '], ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
               ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '], ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
               ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '], ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ']]

    # set up piece representations
    pieces = dict(
        [(0, "K+"), (1, "Q+"), (2, "R+"), (3, "B+"), (4, "N+"), (5, "P+"), (6, "K-"), (7, "Q-"), (8, "R-"),
         (9, "B-"), (10, "N-"), (11, "P-")])

    # transcribe board tensor
    for channel in range(0, 12):
        for x in range(0, 8):
            for y in range(0, 8):
                if board[channel][y][x] == 1:
                    display[y][x] = pieces[channel]

    print('')
    if board[13][0][0] == 1:
        print("Black to move")
    else:
        print("White to move")
    print('')

    print('8', end='  ')
    print(display[0])
    print('7', end='  ')
    print(display[1])
    print('6', end='  ')
    print(display[2])
    print('5', end='  ')
    print(display[3])
    print('4', end='  ')
    print(display[4])
    print('3', end='  ')
    print(display[5])
    print('2', end='  ')
    print(display[6])
    print('1', end='  ')
    print(display[7])
    print('')
    print(' ', end='  ')
    print(['a ', 'b ', 'c ', 'd ', 'e ', 'f ', 'g ', 'h '])
    print('')

### data recovery:
#data = td.TrainingData(r'D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMs.pgn')

# with open(r'D:\Machine Learning\DeepLearningChessAI\Data\full_dataset.db', 'rb') as file:
#     data = pickle.load(file)



### converting to string rep of board for key
"""
with open(r'D:\Machine Learning\DeepLearningChessAI\full_dataset.db', 'rb') as file:
    data = pickle.load(file)

corrupt = list()
for index in range(0, len(data)):
    stringBoard = boardToString(data[index][0])
    if stringBoard != None:
        data[index] = (stringBoard, data[index][1])
    else:
        corrupt.append(index)

# remove bad data (reversed to preserve indexes)
print(len(corrupt))
print(corrupt)

for badDatum in reversed(corrupt):
    del data[badDatum]

with open(r'D:\Machine Learning\DeepLearningChessAI\Data\StringKey.db', 'wb') as file:
    pickle.dump(data, file)
"""


### counting / converting test data

## testing a test

start = initialBoard()
plz = torch.equal( start, initialBoard())
print(plz)

with open(r'D:\Machine Learning\DeepLearningChessAI\Data\full_dataset2.db', 'rb') as file:
    data = pickle.load(file)

starting = initialBoard()

print(data[0])

for item in data:
    displayBoard(item[0])
    input()

"""
data = probability(data)

#print(data[0])

with open(r'D:\Machine Learning\DeepLearningChessAI\Data\StringKeyProb.db', 'wb') as file:
    pickle.dump(data, file)
"""

### checking kvps of table
"""
with open(r'D:\Machine Learning\DeepLearningChessAI\Data\StringKeyProb_table.db', 'rb') as file:
    data = pickle.load(file)

start = boardToString(initialBoard())
print(start)
print(data[start])              # openning position NOT in dataset...???***

index = 0
for key in data.keys():
    print(key)
    print(data[key])
    index += 1
    input()

"""

"""
"""
#     print(datum[1])
#     print(type(datum[1]))
#     if datum[1] != 0.0 and datum[1] != 0.5 and datum[1] != 1.0:
#         print(datum)
#         input()

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

