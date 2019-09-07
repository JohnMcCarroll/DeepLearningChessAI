import os
import re
import torch
import numpy as np

"""
    TrainingData
    This class is to handle the data. Will read PGN files and populate hash table(s) with valid games' board positions
    (represented in tensor form) paired with expected predictions (result of the game). Will randomly order training examples 
    and form training and cross validation (and test?) sets.
"""

class TrainingData:
    def __init__(self, dataLocation):
        self.dataset = dict()
        # Populate list of filenames in PGN directory, can skip step later (straight to reading files)
        for filename in os.listdir(dataLocation):
            file = open(dataLocation + "\\" + filename, 'r', 1)
            result = ""
            for line in file:
                fields = line.split(" ")
                if fields[0] == "1.":
                    # filter games decided by time or disconnection
                    if re.match(fields[len(fields) - 4], "forfeits"):
                        print("invalid game")
                        
                    else:
                        # store result of game (1 = white wins, 0 = white loses, 1/2 = draw)
                        result = fields[len(fields) - 1].split("-")[0]

                        # parse game moves and pair with result
                        # initialize board state tensor
                        board = torch.zeros([2, 8, 8])
                        board[0][0][0] = -5
                        board[0][0][7] = -5
                        board[0][0][1] = -3
                        board[0][0][6] = -3
                        board[0][0][2] = -4
                        board[0][0][5] = -4
                        board[0][0][3] = -9
                        board[0][0][4] = -10
                        board[0][1][:] = -1
                        board[0][7][0] = 5
                        board[0][7][7] = 5
                        board[0][7][1] = 3
                        board[0][7][6] = 3
                        board[0][7][2] = 4
                        board[0][7][5] = 4
                        board[0][7][3] = 9
                        board[0][7][4] = 10
                        board[0][6][:] = 1
                        board[1][:][:] = 1

                        self.dataset[board] = result        #store first board state

                        # piece type hashtable
                        types = dict()
                        types['K'] = 10
                        types['Q'] = 9
                        types['R'] = 5
                        types['B'] = 4
                        types['N'] = 3

                        # initalize move variables
                        moveRow = 0
                        moveCol = 0
                        pieceType = 0
                        pieceLoc = 0
                        prevField = '1.'
                        promotion = False

                        for field in fields:
                            # TESTING: print initial board state:
                            print('Before Move:')
                            print(board)

                            # remove checks
                            field = field.strip('+#') 

                            # color determination
                            if prevField[-1] == '.':
                                color = 1
                            else:
                                color = -1
                            board[1][:][:] = color*(-1)     #opposite piece we are moving because this indicates whose turn it WILL be

                            # move and piece type parsing
                            if field[-1] == '.':
                                pass
                            elif field[0] == '{':
                                break
                            elif len(field) == 2:
                                moveCol = ord(field[0]) - 96
                                moveRow = int(field[1])
                                pieceType = color

                            elif len(field) == 3:
                                if field[0] == 'O':
                                    pieceType = 'King Side Castle'
                                else:
                                    pieceType = types[field[0]] * color
                                    moveCol = ord(field[1]) - 96
                                    moveRow = int(field[2])

                            elif len(field) == 4:
                                if field[1] == 'x':
                                    if ord(field[0]) > 96:
                                        pieceType = color
                                        pieceLoc = ord(field[0]) - 96
                                    else:
                                        pieceType = types[field[0]] * color
                                    moveCol = ord(field[2]) - 96
                                    moveRow = int(field[3])
                                elif field[2] == '=':
                                    promotion = True
                                    pieceType = types[field[3]] * color
                                    moveCol = ord(field[0]) - 96
                                    moveRow = int(field[1])
                                else:
                                    pieceType = types[field[0]] * color
                                    if ord(field[1]) < 58: 
                                        pieceLoc = -int(field[1])
                                    else:
                                        pieceLoc = ord(field[1]) - 96
                                    moveCol = ord(field[2]) - 96
                                    moveRow = int(field[3])

                            elif len(field) == 5:
                                if field[0] == 'O':
                                    pieceType = 'Queen Side Castle'
                                elif field[2] == 'x':
                                    pieceType = types[field[0]] * color
                                    if ord(field[1]) < 58: 
                                            pieceLoc = -int(field[1])
                                        else:
                                            pieceLoc = ord(field[1]) - 96
                                    moveCol = ord(field[3]) - 96
                                    moveRow = int(field[4])

                            elif len(field) == 6:
                                promotion = True
                                pieceType = types[field[5]] * color
                                if ord(field[0]) < 58: 
                                            pieceLoc = -int(field[0])
                                        else:
                                            pieceLoc = ord(field[0]) - 96
                                moveCol = ord(field[2]) - 96
                                moveRow = int(field[3])
                            else:
                                print(field)

                            prevField = field

                            # Alter board state - move piece

                            # CASTLE
                            if isinstance(pieceType, str):
                                if color == 1:
                                    moveRow = 8
                                else:
                                    moveRow = 1

                                if pieceType == "King Side Castle":
                                    board[0][moveRow][4] = 0
                                    board[0][moveRow][7] = 0
                                    board[0][moveRow][6] = 10*color
                                    board[0][moveRow][5] = 5*color
                                else:
                                    board[0][moveRow][4] = 0
                                    board[0][moveRow][0] = 0
                                    board[0][moveRow][2] = 10*color
                                    board[0][moveRow][3] = 5*color

                            # PROMOTION
                            

                            # KING movement
                            elif abs(pieceType) == 10:                        
                                #search for king to be moved and remove piece from square
                                for x in range(-1,2):
                                    for y in range(-1,2):
                                        if board[0][moveRow + x][moveCol + y] == pieceType:
                                            board[0][moveRow + x][moveCol + y] = 0
                                            break
                                #place piece in new square
                                board[0][moveRow][moveCol] = pieceType

                            # KNIGHT movement
                            elif abs(pieceType) == 3:
                                # if specific piece noted, search that row / col
                                if pieceLoc < 0:
                                    pieceLoc = -pieceLoc
                                    x = 0
                                    while board[0][pieceLoc][x] != 3:
                                        x += 1
                                    # remove piece from square & place in new square
                                    board[0][pieceLoc][x] = 0
                                    board[0][moveRow][moveCol] = pieceType
                                elif pieceLoc > 0:
                                    x = 0
                                    while board[0][x][pieceLoc] != 3:
                                        x += 1
                                    # remove piece from square & place in new square
                                    board[0][x][pieceLoc] = 0
                                    board[0][moveRow][moveCol] = pieceType
                                else:                                                                  # this could be optimized, searches everything even if knight found...*
                                    for x in [-2,2]:
                                        for y in [-1,1]:
                                            if board[0][moveRow + x][moveCol + y] == pieceType:
                                                board[0][moveRow + x][moveCol + y] = 0
                                                break
                                    for y in [-2,2]:
                                        for x in [-1,1]:
                                            if board[0][moveRow + x][moveCol + y] == pieceType:
                                                board[0][moveRow + x][moveCol + y] = 0
                                                break
                                    board[0][moveRow][moveCol] = pieceType

                            # BISHOP Movement
                            elif abs(pieceType) == 4:
                                startRow = moveRow
                                startCol = moveCol
                                notFound = True
                                # set up first diagonal search from upper left most square
                                while startRow > 0 and startCol > 0:
                                    startRow -= 1
                                    startCol -= 1
                                while startRow <= 8 and startCol <= 8:
                                    if board[0][startRow][startCol] == pieceType
                                        board[0][startRow][startCol] = 0
                                        notFound = False
                                        break
                                    startRow += 1
                                    startCol += 1
                                # set up second diagonal search from lower left most square
                                startRow = moveRow
                                startCol = moveCol
                                while startRow <= 8 and startCol > 0 and notFound:
                                    startRow += 1
                                    startCol -= 1
                                while startRow >= 0 and startCol <= 8 and notFound:
                                    if board[0][startRow][startCol] == pieceType
                                        board[0][startRow][startCol] = 0
                                        break
                                    startRow -= 1
                                    startCol += 1
                            board[0][moveRow][moveCol] = pieceType

                            # ROOK Movement
                            elif abs(pieceType) == 5:
                                # if specific piece noted, search that row / col
                                if pieceLoc < 0:
                                    pieceLoc = -pieceLoc
                                    x = 0
                                    while board[0][pieceLoc][x] != 3:
                                        x += 1
                                    # remove piece from square & place in new square
                                    board[0][pieceLoc][x] = 0
                                elif pieceLoc > 0:
                                    x = 0
                                    while board[0][x][pieceLoc] != 3:
                                        x += 1
                                    # remove piece from square & place in new square
                                    board[0][x][pieceLoc] = 0
                                else:
                                    notFound = True
                                    # search row first for rook
                                    x = 0
                                    while x <= 8:
                                        if board[0][x][moveCol] == pieceType
                                            board[0][x][moveCol] = 0
                                            notFound = False
                                            break
                                        x += 1
                                    # search column next
                                    y = 0
                                    while y <= 8 and notFound:
                                        if board[0][moveRow][y] == pieceType
                                            board[0][moveRow][y] = 0
                                            break
                                        y += 1
                                board[0][moveRow][moveCol] = pieceType

                            # QUEEN Movement                                                # BUG! scanning board does not take into account obstructing pieces - add condition to check if board not obstructed while searching***
                            elif abs(pieceType) == 9:
                                notFound = True
                                startRow = moveRow
                                startCol = moveCol
                                # set up first diagonal search from upper left most square
                                while startRow > 0 and startCol > 0:
                                    startRow -= 1
                                    startCol -= 1
                                while startRow <= 8 and startCol <= 8:
                                    if board[0][startRow][startCol] == pieceType
                                        board[0][startRow][startCol] = 0
                                        notFound = False
                                        break
                                    startRow += 1
                                    startCol += 1
                                # set up second diagonal search from lower left most square
                                startRow = moveRow
                                startCol = moveCol
                                while startRow <= 8 and startCol > 0 and notFound:
                                    startRow += 1
                                    startCol -= 1
                                while startRow >= 0 and startCol <= 8 and notFound:
                                    if board[0][startRow][startCol] == pieceType
                                        board[0][startRow][startCol] = 0
                                        notFound = False
                                        break
                                    startRow -= 1
                                    startCol += 1
                                # search row first for queen
                                x = 0
                                while x <= 8 and notFound:
                                    if board[0][x][moveCol] == pieceType
                                        board[0][x][moveCol] = 0
                                        notFound = False
                                        break
                                    x += 1
                                # search column next
                                y = 0
                                while y <= 8 and notFound:
                                    if board[0][moveRow][y] == pieceType
                                        board[0][moveRow][y] = 0
                                        notFound = False
                                        break
                                    y += 1
                            board[0][moveRow][moveCol] = pieceType

                            # PAWN Movement
                            elif abs(pieceType) == 1:
                                if pieceLoc == 0:
                                    if board[0][moveRow + color][moveCol] == pieceType:
                                        board[0][moveRow + color][moveCol] = 0
                                    elif board[0][moveRow + color*2][moveCol] == pieceType:
                                        board[0][moveRow + color*2][moveCol] = 0
                                else:
                                    board[0][moveRow + color][pieceLoc] = 0
                                board[0][moveRow][moveCol] = pieceType
                                


                                




                            # TESTING: print resulting board state:
                            print('After Move:')
                            print(board)

                                
                        



# testing
TD = TrainingData("D:\\Machine Learning\\Chess Database\\2000+ Games")