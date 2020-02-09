import torch
import pickle
import copy
import traceback

# import PredictionVisualization
import sys

class Node:
    def __init__(self, boardState, parent=None):
        self.boardState = boardState.cuda()     # the position of the pieces, on GPU
        self.color = ""                         # the color to move

        self.colorChannels = list()             # set of channels that house own color pieces
        self.oppColorChannels = list()          # set of channels that house opponent's color pieces

        self.enPassant = []                     # the coordinates to an opponent's vulnerable
        self.WKC=True                           # White Kingside Castle available
        self.WQC=True                           # White Queenside Castle available
        self.BKC=True                           # Black Kingside Castle available
        self.BQC=True                           # Black Queenside Castle available

        self.parent = parent
        self.children = list()                   # the set of all possible board states after next move

        #self.value = 0                          # cnn eval of node's boardstate                          

        # determine which color's turn
        if self.boardState[13, 0, 0] == 0:
            self.color = "White"
            self.colorChannels = [0, 1, 2, 3, 4, 5]
            self.oppColorChannels = [6, 7, 8, 9, 10, 11]
        else:
            self.color = "Black"
            self.colorChannels = [6, 7, 8, 9, 10, 11]
            self.oppColorChannels = [0, 1, 2, 3, 4, 5]

        # determine historical boolean statuses
        if parent is not None:
            self.updateStatus(parent, self.boardState)

        #debugging***
        # if 0 == torch.nonzero(boardState[self.colorChannels[0], :, :]).nelement():
        #     print("child:")
        #     print(self)
        #     print("parent:")
        #     print(self.parent)
        #     print("grandparent:")
        #     print(self.parent.getParent())
        #     print("great grandparent:")
        #     print(self.parent.getParent().getParent())
        #     print("great great grandparent:")
        #     print(self.parent.getParent().getParent().getParent())

    def createChildren(self):
        # clear children
        self.children = []

        # iterate through each square on board seeing if piece of color resides there

        for channel in self.colorChannels:
            # find piece locations
            locations = torch.nonzero(self.boardState[channel, :, :])

            for coordinates in locations:

                # identify piece and call helper method to generate all legal moves

                if channel % 6 == 0:
                    for move in self.kingMoves(self.boardState, coordinates):
                        self.children.append(Node(move, self))                   
                elif channel % 6 == 1:
                    for move in self.queenMoves(self.boardState, coordinates):
                        self.children.append(Node(move, self))
                elif channel % 6 == 2:
                    for move in self.rookMoves(self.boardState, coordinates):
                        self.children.append(Node(move, self))
                elif channel % 6 == 3:
                    for move in self.bishopMoves(self.boardState, coordinates):
                        self.children.append(Node(move, self))
                elif channel % 6 == 4:
                    for move in self.knightMoves(self.boardState, coordinates):
                        self.children.append(Node(move, self))
                elif channel % 6 == 5:
                    for move in self.pawnMoves(self.boardState, coordinates):
                        self.children.append(Node(move, self))
                else:
                    print(channel)
    
    def kingMoves(self, boardState, coordinates):
        moves = list()
        board = copy.deepcopy(boardState)

        # piece movement
        for row in [-1, 0, 1]:
            for col in [-1, 0, 1]:
                if (row == 0 and col == 0) or coordinates[0] + row < 0 or coordinates[0] + row > 7 or coordinates[1] + col < 0 or coordinates[1] + col > 7:
                    continue    # skip if checking same spot or out of bounds
                else:
                    # check to ensure own color piece not in way
                    if torch.max(board[self.colorChannels, coordinates[0] + row, coordinates[1] + col]) == 0:
                        # clear Channel
                        board[self.colorChannels[0], :, :] = 0
                        # remove captured piece
                        board[0:12, coordinates[0] + row, coordinates[1] + col] = 0
                        # place king
                        board[self.colorChannels[0], coordinates[0] + row, coordinates[1] + col] = 1

                        # make sure legal move
                        if not self.inCheck(board):
                            # change turn
                            self.changeTurn(board)

                            # add possible move to list
                            moves.append(board)

                    # refresh boardState
                    board = copy.deepcopy(boardState)

        # castling
        if self.color == "White":
            if self.WQC:
                if torch.max(board[0:12, 7, 1:4]) < 1 and not self.inCheck(board, [7, 3]) and not self.inCheck(board, [7, 2]):           # check to see if interim squares empty and not under threat, if yes move piece

                    board[0, :, :] = 0
                    board[2, 7, 0] = 0

                    board[0, 7, 2] = 1
                    board[2, 7, 3] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)

                    board = copy.deepcopy(boardState)

            if self.WKC:
                    if torch.max(board[0:12, 7, 5:7]) < 1 and not self.inCheck(board, [7, 5]) and not self.inCheck(board, [7, 6]):           # check to see if interim squares empty, if yes move pieces

                        board[0, :, :] = 0
                        board[2, 7, 7] = 0

                        board[0, 7, 6] = 1
                        board[2, 7, 5] = 1

                        if not self.inCheck(board):

                            # change turn
                            self.changeTurn(board)

                            # add possible move to list
                            moves.append(board)
        else:
            if self.BQC:
                if torch.max(board[0:12, 0, 1:4]) < 1 and not self.inCheck(board, [0, 3]) and not self.inCheck(board, [0, 2]):           # check to see if interim squares empty, if yes move pieces

                    board[6, :, :] = 0
                    board[8, 0, 0] = 0

                    board[6, 0, 2] = 1
                    board[8, 0, 3] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)

                    board = copy.deepcopy(boardState)

            if self.BKC:
                    if torch.max(board[0:12, 0, 5:7]) < 1 and not self.inCheck(board, [0, 5]) and not self.inCheck(board, [0, 6]):           # check to see if interim squares empty, if yes move pieces

                        board[6, :, :] = 0
                        board[8, 0, 7] = 0

                        board[6, 0, 6] = 1
                        board[8, 0, 5] = 1

                        if not self.inCheck(board):
                            # change turn
                            self.changeTurn(board)

                            # add possible move to list
                            moves.append(board)

        return moves

    def queenMoves(self, boardState, coordinates):
        moves = list()
        
        for move in self.linearMoves(boardState, coordinates, self.colorChannels[1]):
            moves.append(move)

        for move in self.diagonalMoves(boardState, coordinates, self.colorChannels[1]):
            moves.append(move)

        return moves

    def rookMoves(self, boardState, coordinates):
        moves = list()

        for move in self.linearMoves(boardState, coordinates, self.colorChannels[2]):
            moves.append(move)

        return moves

    def bishopMoves(self, boardState, coordinates):
        moves = list()

        for move in self.diagonalMoves(boardState, coordinates, self.colorChannels[3]):
            moves.append(move)

        return moves

    def knightMoves(self, boardState, coordinates):
        moves = list()
        board = copy.deepcopy(boardState)

        for x in [-2,2]:
            for y in [-1,1]:
                if coordinates[0] + x >= 0 and coordinates[0] + x <= 7 and coordinates[1] + y >= 0 and coordinates[1] + y <= 7:
                    # check to ensure own color piece not in way
                    if torch.max(board[self.colorChannels, coordinates[0] + x, coordinates[1] + y]) == 0:
                        # remove knight from square
                        board[self.colorChannels[4], coordinates[0], coordinates[1]] = 0
                        # clear square to be moved onto
                        board[0:12, coordinates[0] + x, coordinates[1] + y] = 0
                        # place knight
                        board[self.colorChannels[4], coordinates[0] + x, coordinates[1] + y] = 1

                        # make sure legal move
                        if not self.inCheck(board):
                            # change turn
                            self.changeTurn(board)

                            # add possible move to list
                            moves.append(board)

                    # refresh board
                    board = copy.deepcopy(boardState)

        for y in [-2,2]:
            for x in [-1,1]:
                if coordinates[0] + x >= 0 and coordinates[0] + x <= 7 and coordinates[1] + y >= 0 and coordinates[1] + y <= 7:
                    # check to ensure own color piece not in way
                    if torch.max(board[self.colorChannels, coordinates[0] + x, coordinates[1] + y]) == 0:
                        # remove knight from square
                        board[self.colorChannels[4], coordinates[0], coordinates[1]] = 0
                        # clear square to be moved onto
                        board[0:12, coordinates[0] + x, coordinates[1] + y] = 0
                        # place knight
                        board[self.colorChannels[4], coordinates[0] + x, coordinates[1] + y] = 1

                        # make sure legal move
                        if not self.inCheck(board):
                            # change turn
                            self.changeTurn(board)

                            # add possible move to list
                            moves.append(board)

                    # refresh board
                    board = copy.deepcopy(boardState)

        return moves

    def pawnMoves(self, boardState, coordinates):
        moves = list()
        board = copy.deepcopy(boardState)

        # determine direction
        direction = 1
        if self.color == "White":
            direction = -1

        # forward move, 1 space
        if torch.max(board[0:12, coordinates[0] + direction, coordinates[1]]) == 0:        # if no pieces in way 

            # promotion
            if coordinates[0] + direction == 0 or coordinates[0] + direction == 7:
                # move the pawn
                board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0

                # promote to queen
                board[self.colorChannels[1], coordinates[0] + direction, coordinates[1]] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)

                # reset board
                board = copy.deepcopy(boardState)


                # move the pawn
                board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0

                # promote to knight
                board[self.colorChannels[1], coordinates[0] + direction, coordinates[1]] = 0
                board[self.colorChannels[4], coordinates[0] + direction, coordinates[1]] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)

                # reset board
                board = copy.deepcopy(boardState)

            else:
                # move the pawn
                board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                board[self.colorChannels[5], coordinates[0] + direction, coordinates[1]] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)
                
                # reset board
                board = copy.deepcopy(boardState)

        # forward move, 2 spaces
            if coordinates[0] == (3.5 - 2.5*direction) and torch.max(board[0:12, coordinates[0] + direction*2, coordinates[1]]) == 0:        # if no pieces in way and pawn's first move

                # move the pawn
                board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                board[self.colorChannels[5], coordinates[0] + direction*2, coordinates[1]] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)
                
                # reset board
                board = copy.deepcopy(boardState)

        # captures
            # boundary checks
        if coordinates[1] - 1 > -1:
            if torch.max(board[self.oppColorChannels[:], coordinates[0] + direction, coordinates[1] - 1]) == 1:        # if opp piece to capture

                # promotion
                if coordinates[0] + direction == 0 or coordinates[0] + direction == 7:

                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + direction, coordinates[1] - 1] = 0
                    board[self.colorChannels[1], coordinates[0] + direction, coordinates[1] - 1] = 1        # promote to queen

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)

                    # reset board
                    board = copy.deepcopy(boardState)


                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + direction, coordinates[1] - 1] = 0
                    board[self.colorChannels[4], coordinates[0] + direction, coordinates[1] - 1] = 1        # promote to knight

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)

                    # reset board
                    board = copy.deepcopy(boardState)

                else:
                    
                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + direction, coordinates[1] - 1] = 0
                    board[self.colorChannels[5], coordinates[0] + direction, coordinates[1] - 1] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)

        if coordinates[1] + 1 < 8:
            if torch.max(board[self.oppColorChannels[:], coordinates[0] + direction, coordinates[1] + 1]) == 1:        # if opp piece to capture

                # promotion
                if coordinates[0] + direction == 0 or coordinates[0] + direction == 7:

                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + direction, coordinates[1] + 1] = 0
                    board[self.colorChannels[1], coordinates[0] + direction, coordinates[1] + 1] = 1        # promote to queen

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)

                    # reset board
                    board = copy.deepcopy(boardState)


                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + direction, coordinates[1] + 1] = 0
                    board[self.colorChannels[4], coordinates[0] + direction, coordinates[1] + 1] = 1        # promote to knight

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)

                    # reset board
                    board = copy.deepcopy(boardState)

                else:
                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + direction, coordinates[1] + 1] = 0
                    board[self.colorChannels[5], coordinates[0] + direction, coordinates[1] + 1] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)

        # enPassant Capture
        if self.enPassant:
            if coordinates[1] - 1 == self.enPassant[1] and coordinates[0] + direction == self.enPassant[0]:         # if en passant coordinates in attack range

                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, self.enPassant[0] - direction, self.enPassant[1]] = 0
                    board[self.colorChannels[5], self.enPassant[0], self.enPassant[1]] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)

            if coordinates[1] + 1 == self.enPassant[1] and coordinates[0] + direction == self.enPassant[0]:         # if en passant coordinates in attack range

                    # move the pawn
                    board[self.colorChannels[5], coordinates[0], coordinates[1]] = 0
                    board[0:12, self.enPassant[0] - direction, self.enPassant[1]] = 0
                    board[self.colorChannels[5], self.enPassant[0], self.enPassant[1]] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)

        return moves

    def linearMoves(self, boardState, coordinates, channel):
        moves = list()
        board = copy.deepcopy(boardState)
        notCapture = True

        # upwards file
        for row in range(coordinates[0] - 1, -1, -1):
            if torch.max(board[self.colorChannels[:], row, coordinates[1]]) == 0 and notCapture:        # if no same color pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(board[0:12, row, coordinates[1]]) == 1:
                    notCapture = False

                # move the piece
                board[0:12, coordinates[0], coordinates[1]] = 0
                board[0:12, row, coordinates[1]] = 0
                board[channel, row, coordinates[1]] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)
                
                # reset board
                board = copy.deepcopy(boardState)
            else:
                break

        notCapture = True
        
        # downwards file
        for row in range(coordinates[0] + 1, 8):
            if torch.max(board[self.colorChannels[:], row, coordinates[1]]) == 0 and notCapture:        # if no pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(board[0:12, row, coordinates[1]]) == 1:
                    notCapture = False

                # move the piece
                board[0:12, coordinates[0], coordinates[1]] = 0
                board[0:12, row, coordinates[1]] = 0
                board[channel, row, coordinates[1]] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)
                
                # reset board
                board = copy.deepcopy(boardState)
            else:
                break

        notCapture = True
        
        # left rank
        for col in range(coordinates[1] - 1, -1, -1):
            if torch.max(board[self.colorChannels[:], coordinates[0], col]) == 0 and notCapture:        # if no pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(board[0:12, coordinates[0], col]) == 1:
                    notCapture = False

                # move the piece
                board[0:12, coordinates[0], coordinates[1]] = 0
                board[0:12, coordinates[0], col] = 0
                board[channel, coordinates[0], col] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)
                
                # reset board
                board = copy.deepcopy(boardState)
            else:
                break

        notCapture = True

        # right rank
        for col in range(coordinates[1] + 1, 8):
            if torch.max(board[self.colorChannels[:], coordinates[0], col]) == 0 and notCapture:        # if no pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(board[0:12, coordinates[0], col]) == 1:
                    notCapture = False

                # move the piece
                board[0:12, coordinates[0], coordinates[1]] = 0
                board[0:12, coordinates[0], col] = 0
                board[channel, coordinates[0], col] = 1

                if not self.inCheck(board):
                    # change turn
                    self.changeTurn(board)

                    # add possible move to list
                    moves.append(board)
                
                # reset board
                board = copy.deepcopy(boardState)
            else:
                break

        return moves

    def diagonalMoves(self, boardState, coordinates, channel):
        moves = list()
        board = copy.deepcopy(boardState)
        notCapture = True

        # establish edge proximity
        left = coordinates[1]       
        right = 7 - coordinates[1]
        top = coordinates[0]
        bottom = 7 - coordinates[0]

            # upward left
        distance = top
        if left < top:
            distance = left

        if distance > -1:
            for x in range(1, distance + 1):

                if torch.max(board[self.colorChannels[:], coordinates[0] - x, coordinates[1] - x]) == 0 and notCapture:        # if no same color pieces in way and did not previously capture a piece

                    # check to see if capturing an opponent's piece
                    if torch.max(board[0:12, coordinates[0] - x, coordinates[1] - x]) == 1:
                        notCapture = False

                    # move the piece
                    board[0:12, coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] - x, coordinates[1] - x] = 0
                    board[channel, coordinates[0] - x, coordinates[1] - x] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)
                else:
                    break

            notCapture = True

            # upward right
        distance = top
        if right < top:
            distance = right

        if distance > -1:
            for x in range(1, distance + 1):

                if torch.max(board[self.colorChannels[:], coordinates[0] - x, coordinates[1] + x]) == 0 and notCapture:        # if no same color pieces in way and did not previously capture a piece

                    # check to see if capturing an opponent's piece
                    if torch.max(board[0:12, coordinates[0] - x, coordinates[1] + x]) == 1:
                        notCapture = False

                    # move the piece
                    board[0:12, coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] - x, coordinates[1] + x] = 0
                    board[channel, coordinates[0] - x, coordinates[1] + x] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)
                else:
                    break

            notCapture = True

            # downward left
        distance = bottom
        if left < bottom:
            distance = left

        if distance > -1:
            for x in range(1, distance + 1):

                if torch.max(board[self.colorChannels[:], coordinates[0] + x, coordinates[1] - x]) == 0 and notCapture:        # if no same color pieces in way and did not previously capture a piece

                    # check to see if capturing an opponent's piece
                    if torch.max(board[0:12, coordinates[0] + x, coordinates[1] - x]) == 1:
                        notCapture = False

                    # move the piece
                    board[0:12, coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + x, coordinates[1] - x] = 0
                    board[channel, coordinates[0] + x, coordinates[1] - x] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)
                else:
                    break

            notCapture = True

            # downward right
        distance = bottom
        if right < bottom:
            distance = right

        if distance > -1:
            for x in range(1, distance + 1):

                if torch.max(board[self.colorChannels[:], coordinates[0] + x, coordinates[1] + x]) == 0 and notCapture:        # if no same color pieces in way and did not previously capture a piece

                    # check to see if capturing an opponent's piece
                    if torch.max(board[0:12, coordinates[0] + x, coordinates[1] + x]) == 1:
                        notCapture = False

                    # move the piece
                    board[0:12, coordinates[0], coordinates[1]] = 0
                    board[0:12, coordinates[0] + x, coordinates[1] + x] = 0
                    board[channel, coordinates[0] + x, coordinates[1] + x] = 1

                    if not self.inCheck(board):
                        # change turn
                        self.changeTurn(board)

                        # add possible move to list
                        moves.append(board)
                    
                    # reset board
                    board = copy.deepcopy(boardState)
                else:
                    break

        return moves

    def inCheck(self, boardState=1, coordinates=[-1,-1]):
        check = False

        # handle defaults
        if type(boardState) == int:
            boardState = self.boardState

        if coordinates == [-1, -1]:
            # get king's coordinates if no square specified
            try:
                coordinates = torch.nonzero(boardState[self.colorChannels[0], :, :])[0]             # ***indexing issue - nonzero returned no coords... no king on board?***
            except IndexError:
                sys.exit()

        # dummy loop to enable skipping
        for dummy in range(0,1):
            # king
            for row in [-1, 0, 1]:
                for col in [-1, 0, 1]:
                    if (row == 0 and col == 0) or coordinates[0] + row < 0 or coordinates[0] + row > 7 or coordinates[1] + col < 0 or coordinates[1] + col > 7:
                        continue    # skip if checking same spot or out of bounds
                    else:      
                        if boardState[self.oppColorChannels[0], coordinates[0] + row, coordinates[1] + col] == 1:          #check if opp king in radius of king
                            check = True
                            break

            # pawns
            direction = 1
            if self.color == "White":
                direction = -1
            
            # make sure not out of bounds
            if coordinates[0] + direction < 8 and coordinates[0] + direction > -1:

                if coordinates[1] - 1 > -1:
                    if boardState[self.oppColorChannels[5], coordinates[0] + direction, coordinates[1] - 1]:
                        check = True
                        break

                if coordinates[1] + 1 < 8:
                    if boardState[self.oppColorChannels[5], coordinates[0] + direction, coordinates[1] + 1]:
                        check = True
                        break

            # knights
            for x in [-2,2]:
                for y in [-1,1]:

                    # bounds check
                    if coordinates[0] + x < 8 and coordinates[0] + x > -1 and coordinates[1] + y < 8 and coordinates[1] + y > -1:
                            
                        if boardState[self.oppColorChannels[4], coordinates[0] + x, coordinates[1] + y]:
                            check = True
                            break

            for y in [-2,2]:
                for x in [-1,1]:

                     # bounds check
                    if coordinates[0] + x < 8 and coordinates[0] + x > -1 and coordinates[1] + y < 8 and coordinates[1] + y > -1:
                            
                        if boardState[self.oppColorChannels[4], coordinates[0] + x, coordinates[1] + y]:
                            check = True
                            break

            if check:
                break

            # rooks & queens
                # up
            for row in range(coordinates[0] - 1, -1, -1):
                # check if in line of opp rook or queen
                if torch.max(boardState[self.oppColorChannels[1:3], row, coordinates[1]]):
                    check = True
                    break
                # check if piece in way
                if torch.max(boardState[0:12, row, coordinates[1]]):
                    break

                # down
            for row in range(coordinates[0] + 1, 8):
                # check if in line of opp rook or queen
                if torch.max(boardState[self.oppColorChannels[1:3], row, coordinates[1]]):
                    check = True
                    break
                # check if piece in way
                if torch.max(boardState[0:12, row, coordinates[1]]):
                    break

                # left
            for col in range(coordinates[1] - 1, -1, -1):
                # check if in line of opp rook or queen
                if torch.max(boardState[self.oppColorChannels[1:3], coordinates[0], col]):
                    check = True
                    break
                # check if piece in way
                if torch.max(boardState[0:12, coordinates[0], col]):
                    break

                # right
            for col in range(coordinates[1] + 1, 8):
                # check if in line of opp rook or queen
                if torch.max(boardState[self.oppColorChannels[1:3], coordinates[0], col]):
                    check = True
                    break
                # check if piece in way
                if torch.max(boardState[0:12, coordinates[0], col]):
                    break

            if check:
                break

            # bishops and queens
                # establish edge proximity
            left = coordinates[1] - 1           # minus one because we start search in front of king's square
            right = 6 - coordinates[1]
            top = coordinates[0] - 1
            bottom = 6 - coordinates[0]

                # upward left
            distance = top
            if left < top:
                distance = left

            if distance > -1:
                for x in range(1, distance + 2):
                    # check if in line of opp bish or queen
                    if torch.max(boardState[self.oppColorChannels[1], coordinates[0] - x, coordinates[1] - x]) or torch.max(boardState[self.oppColorChannels[3], coordinates[0] - x, coordinates[1] - x]):
                        check = True
                        break
                    # check if piece in way
                    if torch.max(boardState[0:12, coordinates[0] - x, coordinates[1] - x]):
                        break

                # upward right
            distance = top
            if right < top:
                distance = right

            if distance > -1:
                for x in range(1, distance + 2):
                    # check if in line of opp bish or queen
                    if torch.max(boardState[self.oppColorChannels[1], coordinates[0] - x, coordinates[1] + x]) or torch.max(boardState[self.oppColorChannels[3], coordinates[0] - x, coordinates[1] + x]):
                        check = True
                        break
                    # check if piece in way
                    if torch.max(boardState[0:12, coordinates[0] - x, coordinates[1] + x]):
                        break

                # downward left
            distance = bottom
            if left < bottom:
                distance = left

            if distance > -1:
                for x in range(1, distance + 2):
                    # check if in line of opp bish or queen
                    if torch.max(boardState[self.oppColorChannels[1], coordinates[0] + x, coordinates[1] - x]) or torch.max(boardState[self.oppColorChannels[3], coordinates[0] + x, coordinates[1] - x]):
                        check = True
                        break
                    # check if piece in way
                    if torch.max(boardState[0:12, coordinates[0] + x, coordinates[1] - x]):
                        break

                # downward right
            distance = bottom
            if right < bottom:
                distance = right

            if distance > -1:
                for x in range(1, distance + 2):
                    # check if in line of opp bish or queen
                    if torch.max(boardState[self.oppColorChannels[1], coordinates[0] + x, coordinates[1] + x]) or torch.max(boardState[self.oppColorChannels[3], coordinates[0] + x, coordinates[1] + x]):
                        check = True
                        break
                    # check if piece in way
                    if torch.max(boardState[0:12, coordinates[0] + x, coordinates[1] + x]):
                        break
        
        return check

    def changeTurn(self, board):
        if self.color == "White":
            board[12:14, :, :] = 1
        else:
            board[12:14, :, :] = 0

    def getChildren(self):
        return self.children

    def getChild(self, index):
        return self.children[index]

    def __str__(self):
        string = "\n"

        # create display
        display = [['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  '], ['  ','  ','  ','  ','  ','  ','  ','  ']]

        # set up piece representations
        pieces = dict([ (0, "K+"), (1, "Q+"), (2, "R+"), (3, "B+"), (4, "N+"), (5, "P+"), (6, "K-"), (7, "Q-"), (8, "R-"), (9, "B-"), (10, "N-"), (11, "P-") ])

        # transcribe board tensor
        for channel in range(0,12):
            for x in range(0,8):
                for y in range(0,8):
                    if self.boardState[channel][y][x] == 1:
                        display[y][x] = pieces[channel]

        #print(self.boardState)

        if self.boardState[13][0][0] == 1:
            string = string + "Black to move\n\n"
        else:
            string = string + "White to move\n\n"
        
        string = string + "8  " + display[0].__str__() + "\n"
        string = string + "7  " + display[1].__str__() + "\n"
        string = string + "6  " + display[2].__str__() + "\n"
        string = string + "5  " + display[3].__str__() + "\n"
        string = string + "4  " + display[4].__str__() + "\n"
        string = string + "3  " + display[5].__str__() + "\n"
        string = string + "2  " + display[6].__str__() + "\n"
        string = string + "1  " + display[7].__str__() + "\n\n"
        string = string + "   " + ['a ', 'b ', 'c ', 'd ', 'e ', 'f ', 'g ', 'h '].__str__() + "\n"

        return string

    def getBoard(self):
        return self.boardState

    def updateStatus(self, parent, board):
        # copy parent's status
        parentStat = parent.getStatus()
        self.WKC = parentStat[0]
        self.WQC = parentStat[1]
        self.BKC = parentStat[2]
        self.BQC = parentStat[3]

        # update status based on last move
        parentBoard = parent.getBoard()

            # en Passant
        pawnLocs = torch.nonzero(parentBoard[self.oppColorChannels[5], :, :])

        for coordinates in pawnLocs:
            if coordinates[0] == 6 and self.color == "Black":
                if self.boardState[self.oppColorChannels[5], coordinates[0], coordinates[1]] == 0 and self.boardState[self.oppColorChannels[5], coordinates[0] - 2, coordinates[1]] == 1:       #if pawn was moved up two squares
                    self.enPassant = [coordinates[0] - 1, coordinates[1]]
                    break

            if coordinates[0] == 1 and self.color == "White":
                if self.boardState[self.oppColorChannels[5], coordinates[0], coordinates[1]] == 0 and self.boardState[self.oppColorChannels[5], coordinates[0] + 2, coordinates[1]] == 1:
                    self.enPassant = [coordinates[0] + 1, coordinates[1]]
                    break


            # castling
        if self.color == "Black":
            # check if white's castling status changed
            if self.WKC or self.WQC:
                if self.boardState[self.oppColorChannels[0], 7, 4] == 0:   #if king moved from starting square
                    self.WKC = False
                    self.WQC = False

            if self.WKC:
                if self.boardState[self.oppColorChannels[2], 7, 7] == 0:   #if kingside rook moved from starting square
                    self.WKC = False

            if self.WQC:
                if self.boardState[self.oppColorChannels[2], 7, 0] == 0:   #if queenside rook moved from starting square
                    self.WQC = False

        else:
            # check if black's castling status changed
            if self.BKC or self.BQC:
                if self.boardState[self.oppColorChannels[0], 0, 4] == 0:   #if king moved from starting square
                    self.BKC = False
                    self.BQC = False

            if self.BKC:
                if self.boardState[self.oppColorChannels[2], 0, 7] == 0:   #if kingside rook moved from starting square
                    self.BKC = False

            if self.BQC:
                if self.boardState[self.oppColorChannels[2], 0, 0] == 0:   #if queenside rook moved from starting square
                    self.BQC = False

    def getStatus(self):
        return (self.WKC, self.WQC, self.BKC, self.BKC)

    def getParent(self):
        return self.parent

# TESTING & DEBUGGING

    #with open(r'D:\Machine Learning\DeepLearningChessAI\small_val_set.db', 'rb') as file:
    #    val_set = pickle.load(file)

#testBoard = val_set[1][0]

# move white pawn to be in capture range from black pawns
#testBoard[0:12, 5, 3] = 0
#testBoard[5, 4, 3] = 1
#testBoard[0:12, 3, 2] = 0
#testBoard[11, 4, 3] = 1

#testBoard[12:14, :, :] = 0

#print(testBoard)

#node = Node(testBoard)

#node.createChildren()
#childNode = node.getChildren().pop()
#print("childNode:")
#print(len(node.getChildren()))
#childNode.createChildren()

#for child in childNode.getChildren():
#    print(child)
#    print("Status:")
#    print(child.getStatus())



            # break out linear & diag movement into own functions to reduce duplicate code {done}}}
            # expand inCheck method function to check if ANY given square is under attack {done}}}
            # fix pass by reference issue with more deepcopies {done}}}
            # change whose turn it is {done}}}
            # bug: spontaneous bishop generation {done}}}
            # bug: no knight moves {done}}}
            # bug: moving a second piece, but same color {done}}}
            # bug: no pawn captures? {done}}}
            # implement rook and king movement flags to help with castling rules / logic {done}}}
            # implement en passant variable that will hold coordinates of vulnerable square for one turn after double pawn move {done}}}

            # test: pawn promotion, castling, en passant, isolated piece moves?
            # capture promotions
# 3 fold repeat? 30 move draw? -> should those responsibilities lie in Player?... probably b/c involves game states