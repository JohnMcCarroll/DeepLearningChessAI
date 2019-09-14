import os
import re
import torch
import numpy as np
import time

"""
    TrainingData
    This class is to handle the data. Will read PGN files and populate hash table(s) with valid games' board positions
    (represented in tensor form) paired with expected predictions (result of the game). Will randomly order training examples 
    and form training and cross validation (and test?) sets.
"""

class TrainingData:
    def __init__(self, filePath):
        # open training data file
        self.dataset = dict()
        file = open(filePath, 'r', 1, encoding='utf-8')
        result = ""
        meetsCriteria = False

        # parse through file, collecting positions / result from games that meet criteria (no time wins...)
        for line in file:
            fields = line.split(" ")
            if fields[0] == "[Termination" and not re.search("time", line) and not re.search("abandoned", line):
                meetsCriteria = True

            if fields[0] == "1." and meetsCriteria:
                # reset criteria filter
                meetsCriteria = False
                
                # store result of game (1 = white wins, 0 = white loses, 1/2 = draw)
                line = re.split(r'\d\[', line)[0]       # cleaning up line
                fields = line.split(" ")
                result = fields[-1].split("-")[0]

                # parse game moves and pair with result
                # initialize board state tensor
                board = torch.zeros([14, 8, 8])
                # White King
                board[0][7][4] = 1
                # White Queen
                board[1][7][3] = 1
                # White Rooks
                board[2][7][0] = 1
                board[2][7][7] = 1
                # White Bishops
                board[3][7][2] = 1
                board[3][7][5] = 1
                # White Knights
                board[4][7][1] = 1
                board[4][7][6] = 1
                # White Pawns
                board[5][6][:] = 1
                # Black King
                board[6][0][4] = 1
                # Black Queen
                board[7][0][3] = 1
                # Black Rooks
                board[8][0][0] = 1
                board[8][0][7] = 1
                # Black Bishops
                board[9][0][2] = 1
                board[9][0][5] = 1
                # Black Knights
                board[10][0][1] = 1
                board[10][0][6] = 1
                # Black Pawns
                board[11][1][:] = 1

                self.dataset[board] = result        #store first board state

                # piece type hashtable                                          
                channels = dict()
                channels['WK'] = 0
                channels['WQ'] = 1
                channels['WR'] = 2
                channels['WB'] = 3
                channels['WN'] = 4
                channels['WP'] = 5
                channels['BK'] = 6
                channels['BQ'] = 7
                channels['BR'] = 8
                channels['BB'] = 9
                channels['BN'] = 10
                channels['BP'] = 11

                for field in fields:
                    # filter fields
                    
                    if field[-1] == '.' or field[-1] == '}' or field[0] == '{':
                        prevField = field
                        continue

                    # initalize move variables
                    moveRow = -1
                    moveCol = -1
                    pieceType = ''
                    pieceLoc = -1
                    location = ''
                    promotion = ''

                    # remove checks
                    field = field.strip('+#')

                    # color determination
                    if prevField[-3:-1] == '..':
                        color = "B"
                        board[12:14][:][:] = 0          #opposite piece we are moving because this indicates whose turn it WILL be (0 = white's move, 1 = black's move)
                    elif prevField[-1] == '.':
                        color = "W"
                        board[12:14][:][:] = 1
                

                    # store prevField
                    prevField = field

                    # parse move
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

                    # Alter board state - move piece

                    # CASTLE
                    if len(pieceType) > 2 :
                        if color == "W":
                            moveRow = 7
                            kingChannel = 0
                            rookChannel = 2
                        else:
                            moveRow = 0
                            kingChannel = 6
                            rookChannel = 8

                        if pieceType == "King Side Castle":
                            board[kingChannel][moveRow][4] = 0
                            board[rookChannel][moveRow][7] = 0
                            board[kingChannel][moveRow][6] = 1
                            board[rookChannel][moveRow][5] = 1
                        else:
                            board[kingChannel][moveRow][4] = 0
                            board[rookChannel][moveRow][0] = 0
                            board[kingChannel][moveRow][2] = 1
                            board[rookChannel][moveRow][3] = 1

                    # KING movement
                    elif pieceType[1] == "K":
                        # clear Channel
                        board[channels[pieceType]][:][:] = 0
                        # remove captured piece
                        board[:][moveRow][moveCol] = 0
                        # place king
                        board[channels[pieceType]][moveRow][moveCol] = 1
                        
                    # KNIGHT movement
                    elif pieceType[1] == "N":
                        # remove captured piece
                        board[:][moveRow][moveCol] = 0

                        # if specific piece noted, search that row / col
                        if location == 'row':
                            # clear specified row
                            board[channels[pieceType]][pieceLoc][:] = 0
                            
                        elif location == 'col':
                            # clear specified column
                            board[channels[pieceType]][:][pieceLoc] = 0
                            
                        else:                                                                  # this could be optimized, sets everything to zero***
                            for x in [-2,2]:
                                for y in [-1,1]:
                                    if moveRow + x >= 0 and moveRow + x <= 7 and moveCol + y >= 0 and moveCol + y <= 7:
                                        board[channels[pieceType]][moveRow + x][moveCol + y] = 0
                            for y in [-2,2]:
                                for x in [-1,1]:
                                    if moveRow + x >= 0 and moveRow + x <= 7 and moveCol + y >= 0 and moveCol + y <= 7:
                                        board[channels[pieceType]][moveRow + x][moveCol + y] = 0

                        board[channels[pieceType]][moveRow][moveCol] = 1                    

                    # BISHOP Movement
                    elif pieceType[1] == "B":
                        # remove captured piece
                        board[:][moveRow][moveCol] = 0

                        startRow = moveRow
                        startCol = moveCol
                        notFound = True
                        # set up first diagonal search from upper left most square
                        while startRow > 0 and startCol > 0:                                    #optimize later***
                            startRow -= 1
                            startCol -= 1
                        while startRow < 8 and startCol < 8 and notFound:
                            if board[channels[pieceType]][startRow][startCol] == 1:
                                notFound = False

                            board[channels[pieceType]][startRow][startCol] = 0
                            
                            startRow += 1
                            startCol += 1
                        # set up second diagonal search from lower left most square
                        startRow = moveRow
                        startCol = moveCol
                        while startRow < 8 and startCol > 0 and notFound:
                            startRow += 1
                            startCol -= 1
                        while startRow >= 0 and startCol < 8 and notFound:
                            if board[channels[pieceType]][startRow][startCol] == 1:
                                notFound = False
                                
                            board[channels[pieceType]][startRow][startCol] = 0

                            startRow -= 1
                            startCol += 1

                        board[channels[pieceType]][moveRow][moveCol] = 1

                    # ROOK Movement
                    elif pieceType[1] == "R":
                        # remove captured piece
                        board[:][moveRow][moveCol] = 0

                        # if specific piece noted, search that row / col
                        if location == 'row':
                            # clear specified row
                            board[channels[pieceType]][pieceLoc][:] = 0        # negative pieceLoc signifies that it's a row
                            
                        elif location == 'col':
                            # clear specified column
                            board[channels[pieceType]][:][pieceLoc] = 0

                        else:
                            notFound = True
                            # search upper col
                            y = moveRow
                            while y > 0:
                                y -= 1
                                if torch.max(board[:][y][moveCol]) > 0:
                                    if board[channels[pieceType]][y][moveCol] == 1:
                                        board[channels[pieceType]][y][moveCol] = 0
                                        notFound = False
                                        break
                                    else:
                                        break
                                
                            # search lower col
                            y = moveRow
                            while y < 7 and notFound:
                                y += 1
                                if torch.max(board[:][y][moveCol]) > 0:
                                    if board[channels[pieceType]][y][moveCol] == 1:
                                        board[channels[pieceType]][y][moveCol] = 0
                                        notFound = False
                                        break
                                    else:
                                        break

                            # search left row
                            x = moveCol
                            while x > 0 and notFound:
                                x -= 1
                                if torch.max(board[:][moveRow][x]) > 0:
                                    if board[channels[pieceType]][moveRow][x] == 1:
                                        board[channels[pieceType]][moveRow][x] = 0
                                        notFound = False
                                        break
                                    else:
                                        break

                            # search right row
                            x = moveCol
                            while x < 7 and notFound:
                                x += 1
                                if torch.max(board[:][moveRow][x]) > 0:
                                    if board[channels[pieceType]][moveRow][x] == 1:
                                        board[channels[pieceType]][moveRow][x] = 0
                                        notFound = False
                                        break
                                    else:
                                        break

                        board[channels[pieceType]][moveRow][moveCol] = 1

                    # QUEEN Movement  
                    elif pieceType[1] == "Q":
                        # remove captured piece
                        board[:][moveRow][moveCol] = 0

                        # if specific piece noted, clear that row / col
                        if location == 'row':
                            # clear specified row
                            board[channels[pieceType]][pieceLoc][:] = 0        # negative pieceLoc signifies that it's a row
                            
                        elif location == 'col':
                            # clear specified column
                            board[channels[pieceType]][:][pieceLoc] = 0

                        else:
                            notFound = True
                            startRow = moveRow
                            startCol = moveCol

                            # search upper col
                            y = moveRow
                            while y > 0:
                                y -= 1
                                if torch.max(board[:][y][moveCol]) > 0:
                                    if board[channels[pieceType]][y][moveCol] == 1:
                                        board[channels[pieceType]][y][moveCol] = 0
                                        notFound = False
                                        break
                                    else:
                                        break
                            # search lower col
                            y = moveRow
                            while y < 7 and notFound:
                                y += 1
                                if torch.max(board[:][y][moveCol]) > 0:
                                    if board[channels[pieceType]][y][moveCol] == 1:
                                        board[channels[pieceType]][y][moveCol] = 0
                                        notFound = False
                                        break
                                    else:
                                        break
                            # search left row
                            x = moveCol
                            while x > 0 and notFound:
                                x -= 1
                                if torch.max(board[:][moveRow][x]) > 0:
                                    if board[channels[pieceType]][moveRow][x] == 1:
                                        board[channels[pieceType]][moveRow][x] = 0
                                        notFound = False
                                        break
                                    else:
                                        break
                            # search right row
                            x = moveCol
                            while x < 7 and notFound:
                                x += 1
                                if torch.max(board[:][moveRow][x]) > 0:
                                    if board[channels[pieceType]][moveRow][x] == 1:
                                        board[channels[pieceType]][moveRow][x] = 0
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
                                if torch.max(board[:][y][x]) > 0:
                                    if board[channels[pieceType]][y][x] == 1:
                                        board[channels[pieceType]][y][x] = 0
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
                                if torch.max(board[:][y][x]) > 0:
                                    if board[channels[pieceType]][y][x] == 1:
                                        board[channels[pieceType]][y][x] = 0
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
                                if torch.max(board[:][y][x]) > 0:
                                    if board[channels[pieceType]][y][x] == 1:
                                        board[channels[pieceType]][y][x] = 0
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
                                if torch.max(board[:][y][x]) > 0:
                                    if board[channels[pieceType]][y][x] == 1:
                                        board[channels[pieceType]][y][x] = 0
                                        notFound = False
                                        break
                                    else:
                                        break

                        board[channels[pieceType]][moveRow][moveCol] = 1

                    # PAWN Movement & Promotion
                    elif pieceType[1] == "P":
                        # clear captured piece
                        board[:][moveRow][moveCol] = 0      #doesnt work?***

                        if color == "W":
                            direction = 1
                        else:
                            direction = -1

                        if not location:
                            if board[channels[pieceType]][moveRow + direction][moveCol] == 1:
                                board[channels[pieceType]][moveRow + direction][moveCol] = 0
                            elif board[channels[pieceType]][moveRow + direction*2][moveCol] == 1:
                                board[channels[pieceType]][moveRow + direction*2][moveCol] = 0

                        else:
                            board[0][moveRow + direction][pieceLoc] = 0

                        if not promotion:
                            board[channels[pieceType]][moveRow][moveCol] = 1
                        else:
                            board[channels[promotion]][moveRow][moveCol] = 1
                        


                    # TESTING: print resulting board state:
                    print(board)

                                
                        



# testing
TD = TrainingData("D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMs.pgn")