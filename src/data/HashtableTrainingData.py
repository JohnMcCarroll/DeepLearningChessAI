import os
import re
import torch
import torch.utils.data
import numpy as np
import random
import pickle
import joblib
import statistics
import gc


"""
    TrainingData
    This class is to handle the data. Will read PGN files and populate hash table(s) with valid games' board positions
    (represented in tensor form) paired with expected predictions (result of the game). Will randomly order training examples 
    and form training and cross validation (and test?) sets.
"""

class TrainingData (torch.utils.data.Dataset):
    def __init__(self, filePath):
        try:
            # set instance variables
            self.dataset = dict()           # dataset
            self.channels = dict()          # piece type hashtable
            self.channels['WK'] = 0
            self.channels['WQ'] = 1
            self.channels['WR'] = 2
            self.channels['WB'] = 3
            self.channels['WN'] = 4
            self.channels['WP'] = 5
            self.channels['BK'] = 6
            self.channels['BQ'] = 7
            self.channels['BR'] = 8
            self.channels['BB'] = 9
            self.channels['BN'] = 10
            self.channels['BP'] = 11

            # open training data file
            file = open(filePath, 'r', 1, encoding='utf-8')
            result = ""
            meetsCriteria = False
            isStandard = True

            # parse through file, collecting positions & result from games that meet criteria (no time or abandonment wins...)
            for line in file:
                fields = line.split(" ")
                # reset flags between games:
                if fields[0] == "[Site":
                    meetsCriteria = False
                    isStandard = True
                if fields[0] == "[Termination" and not re.search("time", line) and not re.search("abandoned", line):        #check criteria
                    meetsCriteria = True
                
                if fields[0] == "[Variant":
                    isStandard = False
                if fields[0] == "1." and meetsCriteria and isStandard:
                    # reset criteria filter
                    meetsCriteria = False
                    isStandard = True

                    # store result of game (1 = white wins, 0 = white loses, 1/2 = draw)
                    line = re.split(r'\d\[', line)[0]       # cleaning up line
                    fields = line.split(" ")
                    result = fields[-1].split('-')[0]

                    if result == "0":
                        result = 0.0
                    elif result == "1":
                        result = 1.0
                    elif result == "1/2":
                        result = 0.5

                    # parse game moves and pair with result

                    # get initial board state
                    board = self.initialBoard()
                    # self.displayBoard(board)

                    stringBoard = self.boardToString(board)
                    if stringBoard in self.dataset:                                         #store starting board state
                        self.dataset[stringBoard] = self.dataset[stringBoard] + [result]
                    else:
                        self.dataset[stringBoard] = [result]

                    # set prevColor for color determination
                    prevColor = 'B'
                    for field in fields:
                        # filter fields
                        if field[-1] == '.' or field[-1] == '}' or field[0] == '{' or re.search(r'\d\-', field):
                            continue
                        # color determination
                        if prevColor == 'W':
                            color = "B"
                            board[12:14, :, :] = 0          #opposite piece we are moving because this indicates whose turn it WILL be (0 = white's move, 1 = black's move)
                        else:
                            color = "W"
                            board[12:14, :, :] = 1
                    
                        # store prevField
                        prevColor = color
                        # parse move
                        moveRow, moveCol, pieceType, pieceLoc, location, promotion = self.parseMove(field, color)
                        # Alter board state - make move
                        # CASTLE
                        if len(pieceType) > 2 :
                            board = self.castleMove(board, pieceType, color)
                        # KING movement
                        elif pieceType[1] == "K":
                            board = self.kingMove(board, pieceType, moveRow, moveCol)
                        # KNIGHT movement
                        elif pieceType[1] == "N":
                            board = self.knightMove(board, pieceType, moveRow, moveCol, pieceLoc, location)
                        # BISHOP Movement
                        elif pieceType[1] == "B":
                            board = self.bishopMove(board, pieceType, moveRow, moveCol)
                        # ROOK Movement
                        elif pieceType[1] == "R":
                            board = self.rookMove(board, pieceType, moveRow, moveCol, pieceLoc, location)
                        # QUEEN Movement  
                        elif pieceType[1] == "Q":
                            board = self.queenMove(board, pieceType, moveRow, moveCol, pieceLoc, location)
                        # PAWN Movement & Promotion
                        elif pieceType[1] == "P":
                            board = self.pawnMove(board, pieceType, moveRow, moveCol, pieceLoc, location, color, promotion)

                        # change to str board
                        stringBoard = self.boardToString(board)

                        # filter for messed up boards:
                        if stringBoard != None:
                            # store board state
                            if stringBoard in self.dataset:
                                self.dataset[stringBoard] = self.dataset[stringBoard] + [result]
                            else:
                                self.dataset[stringBoard] = [result]

        except Exception as e:
            print("Sample creation failed :(")
            print(e)

    def __getitem__(self, index):
        return self.dataset[index]

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __len__(self):
        return len(self.dataset)

    def displayBoard(self, board):
        # create display
        display = [['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  ']]
        # set up piece representations
        pieces = dict([ (0, "K+"), (1, "Q+"), (2, "R+"), (3, "B+"), (4, "N+"), (5, "P+"), (6, "K-"), (7, "Q-"), (8, "R-"), (9, "B-"), (10, "N-"), (11, "P-") ])
        # transcribe board tensor
        for channel in range(0,12):
            for x in range(0,8):
                for y in range(0,8):
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

    def initialBoard(self):
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

    def parseMove(self, field, color):
        # initalize move variables
        moveRow = -1
        moveCol = -1
        pieceType = ''
        pieceLoc = -1
        location = ''
        promotion = ''
        # remove checks
        field = field.strip('+#')
        # parse move data
        if len(field) == 2:
            moveCol = ord(field[0]) - 97
            moveRow = 8 - int(field[1])
            pieceType = color + "P"
        elif len(field) == 3:
            if field[0] == 'O':
                pieceType = 'King Side Castle'
            else:
                pieceType = color + field[0]
                moveCol = ord(field[1]) - 97
                moveRow = 8 - int(field[2])
        elif len(field) == 4:
            if field[1] == 'x':
                if ord(field[0]) > 96:
                    pieceType = color + "P"
                    pieceLoc = ord(field[0]) - 97
                    location = 'col'
                else:
                    pieceType = color + field[0]
                moveCol = ord(field[2]) - 97
                moveRow = 8 - int(field[3])
            elif field[2] == '=':
                promotion = color + field[3]
                pieceType = color + "P"
                moveCol = ord(field[0]) - 97
                moveRow = 8 - int(field[1])
            else:
                pieceType = color + field[0]
                if ord(field[1]) < 58: 
                    pieceLoc = 8 - int(field[1])
                    location = 'row'
                else:
                    pieceLoc = ord(field[1]) - 97
                    location = 'col'
                moveCol = ord(field[2]) - 97
                moveRow = 8 - int(field[3])
        elif len(field) == 5:
            if field[0] == 'O':
                pieceType = 'Queen Side Castle'
            elif field[2] == 'x':
                pieceType = color + field[0]
                if ord(field[1]) < 58: 
                    pieceLoc = 8 - int(field[1])
                    location = 'row'
                else:
                    pieceLoc = ord(field[1]) - 97
                    location = 'col'
                moveCol = ord(field[3]) - 97
                moveRow = 8 - int(field[4])
        elif len(field) == 6:
            promotion = color + field[5]
            pieceType = color + "P"
            if ord(field[0]) < 58: 
                pieceLoc = 8 - int(field[0])
                location = 'row'
            else:
                pieceLoc = ord(field[0]) - 97
                location = 'col'
            moveCol = ord(field[2]) - 97
            moveRow = 8 - int(field[3])
        else:
            print("parseMove: unidentified field:")
            print(field)
        return moveRow, moveCol, pieceType, pieceLoc, location, promotion
                                
    def castleMove(self, board, pieceType, color):
        if color == "W":
            moveRow = 7
            kingChannel = 0
            rookChannel = 2
        else:
            moveRow = 0
            kingChannel = 6
            rookChannel = 8
        if pieceType == "King Side Castle":
            board[kingChannel, moveRow, 4] = 0
            board[rookChannel, moveRow, 7] = 0
            board[kingChannel, moveRow, 6] = 1
            board[rookChannel, moveRow, 5] = 1
        else:
            board[kingChannel, moveRow, 4] = 0
            board[rookChannel, moveRow, 0] = 0
            board[kingChannel, moveRow, 2] = 1
            board[rookChannel, moveRow, 3] = 1
        return board

    def kingMove(self, board, pieceType, moveRow, moveCol):
        # clear Channel
        board[self.channels[pieceType], :, :] = 0
        # remove captured piece
        board[0:12, moveRow, moveCol] = 0
        # place king
        board[self.channels[pieceType], moveRow, moveCol] = 1
        
        return board

    def knightMove(self, board, pieceType, moveRow, moveCol, pieceLoc, location):
        # remove captured piece
        board[0:12, moveRow, moveCol] = 0
        # if specific piece noted, search that row / col
        if location == 'row':
            # clear specified row
            board[self.channels[pieceType], pieceLoc, :] = 0
            
        elif location == 'col':
            # clear specified column
            board[self.channels[pieceType], :, pieceLoc] = 0
            
        else:                                                                  # this could be optimized, sets everything to zero***
            for x in [-2,2]:
                for y in [-1,1]:
                    if moveRow + x >= 0 and moveRow + x <= 7 and moveCol + y >= 0 and moveCol + y <= 7:
                        board[self.channels[pieceType], moveRow + x, moveCol + y] = 0
            for y in [-2,2]:
                for x in [-1,1]:
                    if moveRow + x >= 0 and moveRow + x <= 7 and moveCol + y >= 0 and moveCol + y <= 7:
                        board[self.channels[pieceType], moveRow + x, moveCol + y] = 0
        board[self.channels[pieceType], moveRow, moveCol] = 1
        return board

    def bishopMove(self, board, pieceType, moveRow, moveCol):
        # remove captured piece
        board[0:12, moveRow, moveCol] = 0
        startRow = moveRow
        startCol = moveCol
        notFound = True
        # set up first diagonal search from upper left most square
        while startRow > 0 and startCol > 0:                                    #optimize later***
            startRow -= 1
            startCol -= 1
        while startRow <= 7 and startCol <= 7 and notFound:
            if board[self.channels[pieceType], startRow, startCol] == 1:
                notFound = False
            board[self.channels[pieceType], startRow, startCol] = 0
            
            startRow += 1
            startCol += 1
        # set up second diagonal search from lower left most square
        startRow = moveRow
        startCol = moveCol
        while startRow < 7 and startCol > 0 and notFound:
            startRow += 1
            startCol -= 1
        while startRow >= 0 and startCol <= 7 and notFound:
            if board[self.channels[pieceType], startRow, startCol] == 1:
                notFound = False
                
            board[self.channels[pieceType], startRow, startCol] = 0
            startRow -= 1
            startCol += 1
        board[self.channels[pieceType], moveRow, moveCol] = 1
        return board

    def rookMove(self, board, pieceType, moveRow, moveCol, pieceLoc, location):
        # remove captured piece
        board[0:12, moveRow, moveCol] = 0
        # if specific piece noted, search that row / col
        if location == 'row':
            # clear specified row
            board[self.channels[pieceType], pieceLoc, :] = 0        # negative pieceLoc signifies that it's a row
            
        elif location == 'col':
            # clear specified column
            board[self.channels[pieceType], :, pieceLoc] = 0
        else:
            notFound = True
            # search upper col
            y = moveRow
            while y > 0:
                y -= 1
                if torch.max(board[0:12, y, moveCol]) > 0:
                    if board[self.channels[pieceType], y, moveCol] == 1:
                        board[self.channels[pieceType], y, moveCol] = 0
                        notFound = False
                        break
                    else:
                        break
                
            # search lower col
            y = moveRow
            while y < 7 and notFound:
                y += 1
                if torch.max(board[0:12, y, moveCol]) > 0:
                    if board[self.channels[pieceType], y, moveCol] == 1:
                        board[self.channels[pieceType], y, moveCol] = 0
                        notFound = False
                        break
                    else:
                        break
            # search left row
            x = moveCol
            while x > 0 and notFound:
                x -= 1
                if torch.max(board[0:12, moveRow, x]) > 0:
                    if board[self.channels[pieceType], moveRow, x] == 1:
                        board[self.channels[pieceType], moveRow, x] = 0
                        notFound = False
                        break
                    else:
                        break
            # search right row
            x = moveCol
            while x < 7 and notFound:
                x += 1
                if torch.max(board[0:12, moveRow, x]) > 0:
                    if board[self.channels[pieceType], moveRow, x] == 1:
                        board[self.channels[pieceType], moveRow, x] = 0
                        notFound = False
                        break
                    else:
                        break
        board[self.channels[pieceType], moveRow, moveCol] = 1
        return board

    def queenMove(self, board, pieceType, moveRow, moveCol, pieceLoc, location):
        # remove captured piece
        board[0:12, moveRow, moveCol] = 0
        # if specific piece noted, clear that row / col
        if location == 'row':
            # clear specified row
            board[self.channels[pieceType], pieceLoc, :] = 0        # negative pieceLoc signifies that it's a row
            
        elif location == 'col':
            # clear specified column
            board[self.channels[pieceType], :, pieceLoc] = 0
        else:
            notFound = True
            startRow = moveRow
            startCol = moveCol
            # search upper col
            y = moveRow
            while y > 0:
                y -= 1
                if torch.max(board[0:12, y, moveCol]) > 0:
                    if board[self.channels[pieceType], y, moveCol] == 1:
                        board[self.channels[pieceType], y, moveCol] = 0
                        notFound = False
                        break
                    else:
                        break
            # search lower col
            y = moveRow
            while y < 7 and notFound:
                y += 1
                if torch.max(board[0:12, y, moveCol]) > 0:
                    if board[self.channels[pieceType], y, moveCol] == 1:
                        board[self.channels[pieceType], y, moveCol] = 0
                        notFound = False
                        break
                    else:
                        break
            # search left row
            x = moveCol
            while x > 0 and notFound:
                x -= 1
                if torch.max(board[0:12, moveRow, x]) > 0:
                    if board[self.channels[pieceType], moveRow, x] == 1:
                        board[self.channels[pieceType], moveRow, x] = 0
                        notFound = False
                        break
                    else:
                        break
            # search right row
            x = moveCol
            while x < 7 and notFound:
                x += 1
                if torch.max(board[0:12, moveRow, x]) > 0:
                    if board[self.channels[pieceType], moveRow, x] == 1:
                        board[self.channels[pieceType], moveRow, x] = 0
                        notFound = False
                        break
                    else:
                        break
            # search upper left diagonal
            y = moveRow
            x = moveCol
            while y > 0 and x > 0 and notFound:
                y -= 1
                x -= 1
                if torch.max(board[0:12, y, x]) > 0:
                    if board[self.channels[pieceType], y, x] == 1:
                        board[self.channels[pieceType], y, x] = 0
                        notFound = False
                        break
                    else:
                        break
            # search upper right diagonal
            y = moveRow
            x = moveCol
            while y > 0 and x < 7 and notFound:
                y -= 1
                x += 1
                if torch.max(board[0:12, y, x]) > 0:
                    if board[self.channels[pieceType], y, x] == 1:
                        board[self.channels[pieceType], y, x] = 0
                        notFound = False
                        break
                    else:
                        break
            # search lower left diagonal
            y = moveRow
            x = moveCol
            while y < 7 and x > 0 and notFound:
                y += 1
                x -= 1
                if torch.max(board[0:12, y, x]) > 0:
                    if board[self.channels[pieceType], y, x] == 1:
                        board[self.channels[pieceType], y, x] = 0
                        notFound = False
                        break
                    else:
                        break
            # search lower right diagonal
            y = moveRow
            x = moveCol
            while y < 7 and x < 7 and notFound:
                y += 1
                x += 1
                if torch.max(board[0:12, y, x]) > 0:
                    if board[self.channels[pieceType], y, x] == 1:
                        board[self.channels[pieceType], y, x] = 0
                        notFound = False
                        break
                    else:
                        break
        board[self.channels[pieceType], moveRow, moveCol] = 1
        return board

    def pawnMove(self, board, pieceType, moveRow, moveCol, pieceLoc, location, color, promotion):
        # clear captured piece
        board[0:12, moveRow, moveCol] = 0
        if color == "W":
            direction = 1
        else:
            direction = -1
        if not location:
            if board[self.channels[pieceType], moveRow + direction, moveCol] == 1:
                board[self.channels[pieceType], moveRow + direction, moveCol] = 0
            elif board[self.channels[pieceType], moveRow + direction*2, moveCol] == 1:
                board[self.channels[pieceType], moveRow + direction*2, moveCol] = 0
        else:
            board[self.channels[pieceType], moveRow + direction, pieceLoc] = 0
        if not promotion:
            board[self.channels[pieceType], moveRow, moveCol] = 1
        else:
            board[self.channels[promotion], moveRow, moveCol] = 1
        return board
    
    # populate cudaDataset with gpu converted dataset
    def cuda(self):
        for datum in self.dataset:
            tensor, result = datum
            self.cudaDataset.append((tensor.cuda(), result))
            print(self.cudaDataset)

    def boardToString(self, board):
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
                print("Failed to convert board to string")
                print(channel)
                return None

        # whose turn?
        turn = board[13, 0, 0]
        if turn == 0:
            string += "W"
        else:
            string += "B"

        return string

    def stringToBoard(self, stringBoard):
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

    def averageResults(self):

        keys = list(self.dataset.keys())

        # calculate probabilities
        for index in keys:
            # calc average
            avg = statistics.mean(self.dataset[index])
            # reassign kvp
            self.dataset[index] = avg

        print("altered data")

    def dictToList(self):
        self.dataset = [(k, v) for k, v in self.dataset.items()]

    def writeDataToFile(self, filepath):
        if type(filepath) is str:
            with open(filepath, 'a') as file:
                for key in self.dataset.keys():
                    file.write(key + " ~ " + str(self.dataset[key]) + "\n")


# quick db size calculator
def quantifyData(data):
    count = 0
    for key in data.keys():
        if type(data[key]) is list:
            count += len(data[key])
            continue
        else:
            count += 1

    return count


# Create TrainingData object, parsing through PGN file
db = TrainingData(r'games.pgn')
gc.collect()
print("done parsing PGN")

print()
print("Size of Dataset before averaging:")
print("Keys:")
print(len(db.dataset.keys()))
print("Items:")
print(quantifyData(db.dataset))

# data alteration
db.averageResults()
gc.collect()
print("averaged results")

print()
print("Size of Dataset after averaging:")
print("Keys:")
print(len(db.dataset.keys()))
print("Items:")
print(quantifyData(db.dataset))


# write hashtable dataset to file
db.writeDataToFile(r'games.txt')
gc.collect()


# UNIFIED DATA:
# decided to store str representation of board in file that's being written. storing each value of tensor is redundant given that
# most values will be zero and will not result in significant reading run time efficiency gain.
# Need:
# - formal method to store data. {done}
# - formal method to read the data and prep it for training loop
# - full dataset stored in one file on HDD {done}
