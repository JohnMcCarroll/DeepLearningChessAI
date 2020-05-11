import torch
import torch.utils.data
import CNN
import pickle
import TrainingData as data

"""
    fun class to visualize board alongside a network's evaluation of the position - manual testing / sanity checking
"""

def displayBoard(board):

    # create display
    display = [['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  ']]

    # set up piece representations
    pieces = dict([ (0, "K+"), (1, "Q+"), (2, "R+"), (3, "B+"), (4, "N+"), (5, "P+"), (6, "K-"), (7, "Q-"), (8, "R-"), (9, "B-"), (10, "N-"), (11, "P-") ])

    # transcribe board tensor
    for channel in range(0,12):
        for x in range(0,8):
            for y in range(0,8):
                if board[0][channel][y][x] == 1:
                    display[y][x] = pieces[channel]

    print('')
    if board[0][13][0][0] == 1:
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

# with open(r'D:\Machine Learning\DeepLearningChessAI\Data\ratioDataset.db', 'rb') as file:
#     train_set = pickle.load(file)

# with open(r'D:\Machine Learning\DeepLearningChessAI\Networks\Statistician.cnn', 'rb') as file:
#     network = pickle.load(file)

# #train_set, validation_set, dummy_set = torch.utils.data.random_split(train_set, [166000, 18000, 72])
# train_loader = torch.utils.data.DataLoader(train_set, 1, shuffle=True)

# for batch in train_loader:

#     board, result = batch
#     displayBoard(board)

#     pred = network(board.cuda())
#     print('Evaluation:')
#     print(pred)

#     input()