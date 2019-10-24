import torch
import pickle
import copy


class Node:
    def __init__(self, boardState):
        self.boardState = boardState            # the position of the pieces
        self.color = ""                         # the color to move
        self.colorChannels = list()
        self.oppColorChannels = list()
        self.children = set()                   # the set of all possible board states after next move

        # determine which color's turn
        if self.boardState[13, 0, 0] == 0:
            self.color = "White"
            self.colorChannels = [0, 1, 2, 3, 4, 5]
            self.oppColorChannels = [6, 7, 8, 9, 10, 11]
        else:
            self.color = "Black"
            self.colorChannels = [6, 7, 8, 9, 10, 11]
            self.oppColorChannels = [0, 1, 2, 3, 4, 5]

    def createChildren(self):
        
        # iterate through each square on board seeing if piece of color resides there

        for channel in self.colorChannels:
            # find piece locations
            locations = torch.nonzero(self.boardState[channel, :, :])

            for coordinates in locations:

                # identify piece and call helper method to generate all legal moves

                if channel % 6 == 0:
                    for move in self.kingMoves(self.boardState, coordinates):       #moves added to children set?
                        self.children.add(Node(move))
                elif channel % 6 == 1:
                    for move in self.queenMoves(self.boardState, coordinates):
                        self.children.add(Node(move))
                elif channel % 6 == 2:
                    for move in self.rookMoves(self.boardState, coordinates):
                        self.children.add(Node(move))
                elif channel % 6 == 3:
                    for move in self.bishopMoves(self.boardState, coordinates):
                        self.children.add(Node(move))
                elif channel % 6 == 4:
                    for move in self.knightMoves(self.boardState, coordinates):
                        self.children.add(Node(move))
                elif channel % 6 == 5:
                    for move in self.pawnMoves(self.boardState, coordinates):
                        self.children.add(Node(move))
                else:
                    print(channel)
    
    def kingMoves(self, boardState, coordinates):
        moves = list()
        board = copy.deepcopy(boardState)

        # piece movement
        for row in [-1, 0, 1]:
            for col in [-1, 0, 1]:
                if (row == 0 and col == 0) or coordinates[0] + row < 0 or coordinates[0] + row > 7 or coordinates[1] + col < 0 or coordinates[1] > 7:
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
        rookLocs = torch.nonzero(board[self.colorChannels[2], :, :])

        if self.color == "White":
            if coordinates[0] == 7 and coordinates[1] == 4: #and rookLocs.count([7,0]):         # MAKE BOOLEAN FLAG FOR HISTORY
                if torch.max(boardState[0:12, 7, 1:4]) < 1:           # check to see if interim squares empty, if yes move pieces
                    
                    #TODO: If no enemy threats in king's path****

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

            if coordinates[0] == 7 and coordinates[1] == 4: #and rookLocs.count([7,7]):
                    if torch.max(board[0:12, 7, 5:7]) < 1:           # check to see if interim squares empty, if yes move pieces

                        #TODO: If no enemy threats in king's path

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
            if coordinates[0] == 0 and coordinates[1] == 4: #and rookLocs.count([0,0]):  
                if torch.max(board[0:12, 0, 1:4]) < 1:           # check to see if interim squares empty, if yes move pieces
                    
                    #TODO: If no enemy threats in king's path

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

            if coordinates[0] == 0 and coordinates[1] == 4: #and rookLocs.count([0,7]):
                    if torch.max(board[0:12, 0, 5:7]) < 1:           # check to see if interim squares empty, if yes move pieces

                        #TODO: If no enemy threats in king's path

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
                print('oy')

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

        

        # TODO: en passant***

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

    def inCheck(self, boardState):
        check = False

        # get king's coordinates
        coordinates = torch.nonzero(boardState[self.colorChannels[0], :, :])[0]

        # dummy loop to enable skipping
        for dummy in range(0,1):
            # king
            if torch.max(boardState[self.oppColorChannels[0], coordinates[0]-1:coordinates[0]+2, coordinates[1]-1:coordinates[1]+2]) == 1:      #check if opp king in radius of king
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
                for x in range(1, distance + 1):
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
                for x in range(1, distance + 1):
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
                for x in range(1, distance + 1):
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
                for x in range(1, distance + 1):
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

    # visibility for testing
    def getChildren(self):
        return self.children

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

with open(r'D:\Machine Learning\DeepLearningChessAI\small_val_set.db', 'rb') as file:
    val_set = pickle.load(file)
    #print(val_set[1][0])

testBoard = val_set[1][0]

# move white pawn to be in capture range from black pawns
testBoard[0:12, 5, 3] = 0
testBoard[5, 4, 3] = 1
#testBoard[0:12, 3, 2] = 0
#testBoard[11, 4, 3] = 1

#testBoard[12:14, :, :] = 0

print(testBoard)

node = Node(testBoard)

node.createChildren()
#childNode = node.getChildren().pop()
#print("childNode:")
print(node)
#childNode.createChildren()

for child in node.getChildren():
    print(child)



            # break out linear & diag movement into own functions to reduce duplicate code {done}}}
# expand inCheck method function to check if ANY given square is under attack
            # fix pass by reference issue with more deepcopies {done}}}
            # change whose turn it is
            # bug: spontaneous bishop generation {done}}}
            # bug: no knight moves {done}}}
            # bug: moving a second piece, but same color {done}}}
            # bug: no pawn captures? {done}}}
# implement rook and king movement flags to help with castling rules / logic
# implement en passant variable that will hold coordinates of vulnerable square for one turn after double pawn move

# test: pawn promotion, castling, en passant, isolated piece moves? 