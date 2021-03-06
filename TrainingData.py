import os
import re
import torch
import torch.utils.data
import numpy as np
import random
import copy

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
            self.dataset = list()           # dataset
            self.cudaDataset = list()       #for gpu processing
            self.channels = dict()      # piece type hashtable
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

            self.probChannels = dict()      # move layer hashtable
            self.probChannels["N"] = 0
            self.probChannels["NE"] = 7
            self.probChannels["E"] = 14
            self.probChannels["SE"] = 21
            self.probChannels["S"] = 28
            self.probChannels["SW"] = 35
            self.probChannels["W"] = 42
            self.probChannels["NW"] = 49
            self.probChannels["knight"] = 56
            self.probChannels["promotion"] = 64


            # open training data file
            file = open(filePath, 'r', 1, encoding='utf-8')
            result = ""
            meetsCriteria = False
            isStandard = True

            # parse through file, collecting positions / result from games that meet criteria (no time or abandonment wins...)
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
                    
                    # interpret result of game (all games start at 0.5, and move incrementally towards win or loss)
                    line = re.split(r'\d\[', line)[0]       # cleaning up line
                    fields = line.split(" ")
                    result = fields[-1].split('-')[0]
                    
                    if result == "0":
                        direction = -1
                    elif result == "1":
                        direction = 1
                    elif result == "1/2":
                        direction = 0

                    result = 0.5

                    # store the number of moves in the game
                    for field in reversed(fields):
                        lastMove = field
                        if lastMove[-1] == ".":
                            break

                    initialLength = len(lastMove)
                    lastMove = lastMove.strip(".")
                    finalLength = len(lastMove)

                    numMoves = int(lastMove)*2

                    if initialLength - finalLength < 3:
                        numMoves+= 1

                    increment = 0.5*(direction / numMoves)

                    # parse game moves and pair with result

                    # get initial board state
                    board = self.initialBoard()
                    previousBoard = copy.deepcopy(board)

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
                    
                        # store prevColor
                        prevColor = color

                        # parse move
                        moveRow, moveCol, pieceType, pieceLoc, location, promotion = self.parseMove(field, color)       ###ADD PROB MATRIX TO RETURN

                        # Alter board state - make move

                        # CASTLE
                        if len(pieceType) > 2 :
                            board, probMatrix = self.castleMove(board, pieceType, color)

                        # KING movement
                        elif pieceType[1] == "K":
                            board, probMatrix = self.kingMove(board, pieceType, moveRow, moveCol)
                            
                        # KNIGHT movement
                        elif pieceType[1] == "N":
                            board, probMatrix = self.knightMove(board, pieceType, moveRow, moveCol, pieceLoc, location)

                        # BISHOP Movement
                        elif pieceType[1] == "B":
                            board, probMatrix = self.bishopMove(board, pieceType, moveRow, moveCol)

                        # ROOK Movement
                        elif pieceType[1] == "R":
                            board, probMatrix = self.rookMove(board, pieceType, moveRow, moveCol, pieceLoc, location)

                        # QUEEN Movement  
                        elif pieceType[1] == "Q":
                            board, probMatrix = self.queenMove(board, pieceType, moveRow, moveCol, pieceLoc, location)

                        # PAWN Movement & Promotion
                        elif pieceType[1] == "P":
                            board, probMatrix = self.pawnMove(board, pieceType, moveRow, moveCol, pieceLoc, location, color, promotion)
                            

                        # store previous board state with result and probabilities
                        self.dataset.append((previousBoard, result, probMatrix))
                        
                        # update previousBoard
                        previousBoard =  copy.deepcopy(board)

                        # update result
                        result += increment

                    # store last move with empty probMatrix
                    self.dataset.append((board, result, self.initialProbMatrix()))

        except Exception as e:
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

    def initialProbMatrix(self):
        return torch.zeros([73, 8, 8])

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
            print(field)
        return moveRow, moveCol, pieceType, pieceLoc, location, promotion
                                
    def castleMove(self, board, pieceType, color):
        probMatrix = self.initialProbMatrix()

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

            # set probability
            probMatrix[self.probChannels["E"] + 1, moveRow, 4] = 1

        else:
            board[kingChannel, moveRow, 4] = 0
            board[rookChannel, moveRow, 0] = 0
            board[kingChannel, moveRow, 2] = 1
            board[rookChannel, moveRow, 3] = 1

            # set probability
            probMatrix[self.probChannels["W"] + 1, moveRow, 4] = 1

        return board, probMatrix

    def kingMove(self, board, pieceType, moveRow, moveCol):
        # init prob
        probMatrix = self.initialProbMatrix()

        # get coordinates
        coordinates = torch.nonzero(self.board[self.channels[pieceType], :, :])
        print(coordinates)

        # identify direction
        direction = ""

        if coordinates[0][0] == moveRow:
            if coordinates[0][1] < moveCol:
                direction = "E"
            else:
                direction = "W"

        if coordinates[0][0] > moveRow:
            if coordinates[0][1] == moveCol:
                direction = "N"
            elif coordinates[0][1] < moveCol:
                direction = "NE"
            else:
                direction = "NW"

        else:
            if coordinates[0][1] == moveCol:
                direction = "S"
            elif coordinates[0][1] < moveCol:
                direction = "SE"
            else:
                direction = "SW"

        probMatrix[self.probChannels[direction], moveRow, moveRow] = 1


        # clear Channel
        board[self.channels[pieceType], :, :] = 0
        # remove captured piece
        board[0:12, moveRow, moveCol] = 0
        # place king
        board[self.channels[pieceType], moveRow, moveCol] = 1
        
        return board, probMatrix

    def knightMove(self, board, pieceType, moveRow, moveCol, pieceLoc, location):
        # set up variables
        moveHeight = 0
        moveWidth = 0
        pieceRow = -1
        pieceCol = -1
        probMatrix = self.initialProbMatrix()

        # remove captured piece
        board[0:12, moveRow, moveCol] = 0

        # if specific piece noted, search that row / col
        if location == 'row':
            # get knight coordinates
            column = torch.nonzero(board[self.channels[pieceType], pieceLoc, :])
            
            # take negative difference
            moveHeight = pieceLoc - moveRow
            moveWidth = column.item() - moveCol

            # store piece location
            pieceRow = pieceLoc
            pieceCol = column.item()

            # clear specified row
            board[self.channels[pieceType], pieceLoc, :] = 0
            
        elif location == 'col':
            # get knight coordinates
            row = torch.nonzero(board[self.channels[pieceType], :, pieceLoc])
            
            # take negative difference
            moveHeight = row.item() - moveRow
            moveWidth = pieceLoc - moveCol

            # store piece location
            pieceRow = row.item()
            pieceCol = pieceLoc

            # clear specified column
            board[self.channels[pieceType], :, pieceLoc] = 0
            
        else:                                                                  # this could be optimized, sets everything to zero***
            for x in [-2,2]:
                for y in [-1,1]:
                    if moveRow + x >= 0 and moveRow + x <= 7 and moveCol + y >= 0 and moveCol + y <= 7:
                        board[self.channels[pieceType], moveRow + x, moveCol + y] = 0

                        # store move info
                        moveHeight = y
                        moveWidth = x
                        pieceRow = moveRow + x
                        pieceCol = moveCol + y

            for y in [-2,2]:
                for x in [-1,1]:
                    if moveRow + x >= 0 and moveRow + x <= 7 and moveCol + y >= 0 and moveCol + y <= 7:
                        board[self.channels[pieceType], moveRow + x, moveCol + y] = 0
                        
                        # store move info
                        moveHeight = y
                        moveWidth = x
                        pieceRow = moveRow + x
                        pieceCol = moveCol + y

        # move the knight
        board[self.channels[pieceType], moveRow, moveCol] = 1

        # log probability
        moveWidth = moveWidth*(-1)
        moveHeight = moveHeight*(-1)

        if moveHeight == -2:
            if moveWidth == 1:
                probMatrix[self.probChannels["knight"], pieceRow, pieceCol] = 1
            elif moveWidth == -1:
                probMatrix[self.probChannels["knight"] + 7, pieceRow, pieceCol] = 1
        elif moveHeight == -1:
            if moveWidth == 2:
                probMatrix[self.probChannels["knight"] + 1, pieceRow, pieceCol] = 1
            elif moveWidth == -2:
                probMatrix[self.probChannels["knight"] + 6, pieceRow, pieceCol] = 1
        elif moveHeight == 1:
            if moveWidth == 2:
                probMatrix[self.probChannels["knight"] + 2, pieceRow, pieceCol] = 1
            elif moveWidth == -2:
                probMatrix[self.probChannels["knight"] + 5, pieceRow, pieceCol] = 1 
        elif moveHeight == 2:
            if moveWidth == 1:
                probMatrix[self.probChannels["knight"] + 3, pieceRow, pieceCol] = 1
            elif moveWidth == -1:
                probMatrix[self.probChannels["knight"] + 4, pieceRow, pieceCol] = 1


        return board, probMatrix

    def bishopMove(self, board, pieceType, moveRow, moveCol):
        # set up variables
        probMatrix = self.initialProbMatrix()
        pieceRow = -1
        pieceCol = -1

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

                # store piece position
                pieceRow = startRow
                pieceCol = startCol

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

                # store piece position
                pieceRow = startRow
                pieceCol = startCol
                
            board[self.channels[pieceType], startRow, startCol] = 0

            startRow -= 1
            startCol += 1

        # move piece
        board[self.channels[pieceType], moveRow, moveCol] = 1

        # set up probability
        distance = abs(pieceCol - moveCol) - 1
        height = moveCol - pieceCol
        width = moveRow - pieceRow

        if height < 0:
            if width > 0:
                probMatrix[self.probChannels["NE"] + distance, moveRow, moveCol] = 1
            else:
                probMatrix[self.probChannels["NW"] + distance, moveRow, moveCol] = 1
        else:
            if width > 0:
                probMatrix[self.probChannels["SE"] + distance, moveRow, moveCol] = 1
            else:
                probMatrix[self.probChannels["SW"] + distance, moveRow, moveCol] = 1


        return board, probMatrix

    def rookMove(self, board, pieceType, moveRow, moveCol, pieceLoc, location):
        # set up variables
        pieceCol = -1
        pieceRow = -1
        probMatrix = self.initialProbMatrix()

        # remove captured piece
        board[0:12, moveRow, moveCol] = 0

        # if specific piece noted, search that row / col
        if location == 'row':
            # save piece location
            pieceRow = pieceLoc
            pieceCol = torch.nonzero(board[self.channels[pieceType], pieceLoc, :]).item()

            # clear specified row
            board[self.channels[pieceType], pieceLoc, :] = 0        # negative pieceLoc signifies that it's a row
            
        elif location == 'col':
            # save piece location
            pieceRow = torch.nonzero(board[self.channels[pieceType], :, pieceLoc]).item()
            pieceCol = pieceLoc

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

                        # store piece location
                        pieceCol = moveCol
                        pieceRow = y
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

                        # store piece location
                        pieceCol = moveCol
                        pieceRow = y
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

                        # store piece location
                        pieceCol = x
                        pieceRow = moveRow
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

                        # store piece location
                        pieceCol = x
                        pieceRow = moveRow
                        break
                    else:
                        break

        # move piece
        board[self.channels[pieceType], moveRow, moveCol] = 1

        # store probability
        moveHeight = moveCol - pieceCol
        moveWidth = moveRow - pieceRow

        if moveWidth == 0:
            if moveHeight < 0:
                # calc distance
                distance = (-1)*moveHeight - 1
                probMatrix[self.probChannels["N"] + distance, pieceRow, pieceCol]
            else:
                # calc distance
                distance = moveHeight - 1
                probMatrix[self.probChannels["S"] + distance, pieceRow, pieceCol]
        elif moveHeight == 0:
            if moveWidth > 0:
                # calc distance
                distance = moveWidth - 1
                probMatrix[self.probChannels["E"] + distance, pieceRow, pieceCol]
            else:
                # calc distance
                distance = (-1)*moveHeight - 1
                probMatrix[self.probChannels["W"] + distance, pieceRow, pieceCol]

        return board, probMatrix

    def queenMove(self, board, pieceType, moveRow, moveCol, pieceLoc, location):
        # set up variables
        # START HERE***

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

# import pickle

db = TrainingData(r'D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMs.pgn')

# with open(r'D:\Machine Learning\DeepLearningChessAI\Data\ratioDataset.db', 'wb') as file:
#     pickle.dump(db.dataset, file)