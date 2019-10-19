import torch
import pickle


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
                    queenMoves(boardState, coordinates)
                elif channel % 6 == 2:
                    rookMoves(boardState, coordinates)
                elif channel % 6 == 3:
                    bishopMoves(boardState, coordinates)
                elif channel % 6 == 4:
                    knightMoves(boardState, coordinates)
                elif channel % 6 == 5:
                    pawnMoves(boardState, coordinates)
                else:
                    print(channel)


        # create a new node to hold boardState

        # add each node to children set...
    
    def kingMoves(self, boardState, coordinates):
        moves = list()
        originalBoard = boardState

        # piece movement
        for row in [-1, 0, 1]:
            for col in [-1, 0, 1]:
                if row == 0 and col == 0:
                    continue    # skip if checking same spot
                else:
                    # check to ensure own color piece not in way
                    if boardState[self.colorChannels, coordinates[0] + row, coordinates[1] + col] == 0:
                        # clear Channel
                        board[self.colorChannels[0], :, :] = 0
                        # remove captured piece
                        board[0:12, coordinates[0] + row, coordinates[1] + col] = 0
                        # place king
                        board[self.colorChannels[0], coordinates[0] + row, coordinates[1] + col] = 1

                        # make sure legal move
                        if not inCheck(boardState):
                            # add possible move to list
                            moves.append(boardState)

                    # refresh boardState
                    boardState = originalBoard

        # castling
        rookLocs = torch.nonzeros(boardState[self.colorChannels[2], :, :])

        if color == "White":
            if coordinates == [7,4] and rookLocs.count([7,0]):  
                if torch.max(boardState[:, 7, 1:4]) < 1:           # check to see if interim squares empty, if yes move pieces
                    
                    #TODO: If no enemy threats in king's path

                    boardState[0, :, :] = 0
                    boardState[2, 7, 0] = 0

                    boardState[0, 7, 2] = 1
                    boardState[2, 7, 3] = 1

                    if not inCheck(boardState):
                        # add possible move to list
                        moves.append(boardState)

                    boardState = originalBoard

            if coordinates == [7,4] and rookLocs.count([7,7]):
                    if torch.max(boardState[:, 7, 5:7]) < 1:           # check to see if interim squares empty, if yes move pieces

                        #TODO: If no enemy threats in king's path

                        boardState[0, :, :] = 0
                        boardState[2, 7, 7] = 0

                        boardState[0, 7, 6] = 1
                        boardState[2, 7, 5] = 1

                        if not inCheck(boardState):
                            # add possible move to list
                            moves.append(boardState)
        else:
            if coordinates == [0,4] and rookLocs.count([0,0]):  
                if torch.max(boardState[:, 0, 1:4]) < 1:           # check to see if interim squares empty, if yes move pieces
                    
                    #TODO: If no enemy threats in king's path

                    boardState[6, :, :] = 0
                    boardState[8, 0, 0] = 0

                    boardState[6, 0, 2] = 1
                    boardState[8, 0, 3] = 1

                    if not inCheck(boardState):
                        # add possible move to list
                        moves.append(boardState)

                    boardState = originalBoard

            if coordinates == [0,4] and rookLocs.count([0,7]):
                    if torch.max(boardState[:, 0, 5:7]) < 1:           # check to see if interim squares empty, if yes move pieces

                        #TODO: If no enemy threats in king's path

                        boardState[6, :, :] = 0
                        boardState[8, 0, 7] = 0

                        boardState[6, 0, 6] = 1
                        boardState[8, 0, 5] = 1

                        if not inCheck(boardState):
                            # add possible move to list
                            moves.append(boardState)

        return moves

    def queenMoves(self, boardState, coordinates):
        return None

    def rookMoves(self, boardState, coordinates):
        moves = list()
        originalBoard = boardState
        notCapture = True

        # upwards file
        for row in range(coordinates[0] - 1, -1, -1):
            if torch.max(boardState[self.colorChannels[:], row, coordinates[1]]) == 0 and notCapture:        # if no pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(boardState[:, row, coordinates[1]]) == 1:
                    notCapture = False

                # move the rook
                boardState[:, coordinates[0], coordinates[1]] = 0
                boardState[:, row, coordinates[1]] = 0
                boardState[self.colorChannels[2], row, coordinates[1]] = 1

                if not inCheck(boardState):
                    # add possible move to list
                    moves.append(boardState)
                
                # reset board
                boardState = originalBoard
            else:
                notCapture = True
                break
        
        # downwards file
        for row in range(coordinates[0] + 1, 8):
            if torch.max(boardState[self.colorChannels[:], row, coordinates[1]]) == 0 and notCapture:        # if no pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(boardState[:, row, coordinates[1]]) == 1:
                    notCapture = False

                # move the rook
                boardState[:, coordinates[0], coordinates[1]] = 0
                boardState[:, row, coordinates[1]] = 0
                boardState[self.colorChannels[2], row, coordinates[1]] = 1

                if not inCheck(boardState):
                    # add possible move to list
                    moves.append(boardState)
                
                # reset board
                boardState = originalBoard
            else:
                notCapture = True
                break
        
        # left rank
        for col in range(coordinates[1] - 1, -1, -1):
            if torch.max(boardState[self.colorChannels[:], coordinates[0], col]) == 0 and notCapture:        # if no pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(boardState[:, coordinates[0], col]) == 1:
                    notCapture = False

                # move the rook
                boardState[:, coordinates[0], coordinates[1]] = 0
                boardState[:, coordinates[0], col] = 0
                boardState[self.colorChannels[2], coordinates[0], col] = 1

                if not inCheck(boardState):
                    # add possible move to list
                    moves.append(boardState)
                
                # reset board
                boardState = originalBoard
            else:
                notCapture = True
                break

        # right rank
        for col in range(coordinates[1] + 1, 8):
            if torch.max(boardState[self.colorChannels[:], coordinates[0], col]) == 0 and notCapture:        # if no pieces in way and did not previously capture a piece

                # check to see if capturing an opponent's piece
                if torch.max(boardState[:, coordinates[0], col]) == 1:
                    notCapture = False

                # move the rook
                boardState[:, coordinates[0], coordinates[1]] = 0
                boardState[:, coordinates[0], col] = 0
                boardState[self.colorChannels[2], coordinates[0], col] = 1

                if not inCheck(boardState):
                    # add possible move to list
                    moves.append(boardState)
                
                # reset board
                boardState = originalBoard
            else:
                notCapture = True
                break

        return moves


    def bishopMoves(self, boardState, coordinates):
        return None

    def knightMoves(self, boardState, coordinates):
        return None

    def pawnMoves(self, boardState, coordinates):
        return None

    def inCheck(self, boardState):
        check = False

        # get king's coordinates
        coordinates = torch.nonzeros(boardState[self.colorChannels[0], :, :])[0]

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

            if boardState[self.oppColorChannels[5], coordinates[0] + direction, coordinates[1] - 1] or boardState[self.oppColorChannels[5], coordinates[0] + direction, coordinates[1] + 1]:
                check = True
                break

            # knights
            for x in [-2,2]:
                for y in [-1,1]:
                    if boardState[self.oppColorChannels[4], coordinates[0] + x, coordinates[1] + y]:
                        check = True
                        break
            for y in [-2,2]:
                for x in [-1,1]:
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
                if torch.max(boardState[:, row, coordinates[1]]):
                    break

                # down
            for row in range(coordinates[0] + 1, 8):
                # check if in line of opp rook or queen
                if torch.max(boardState[self.oppColorChannels[1:3], row, coordinates[1]]):
                    check = True
                    break
                # check if piece in way
                if torch.max(boardState[:, row, coordinates[1]]):
                    break

                # left
            for col in range(coordinates[1] - 1, -1, -1):
                # check if in line of opp rook or queen
                if torch.max(boardState[self.oppColorChannels[1:3], coordinates[0], col]):
                    check = True
                    break
                # check if piece in way
                if torch.max(boardState[:, coordinates[0], col]):
                    break

                # right
            for col in range(coordinates[1] + 1, 8):
                # check if in line of opp rook or queen
                if torch.max(boardState[self.oppColorChannels[1:3], coordinates[0], col]):
                    check = True
                    break
                # check if piece in way
                if torch.max(boardState[:, coordinates[0], col]):
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
                    if torch.max(boardState[:, coordinates[0] - x, coordinates[1] - x]):
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
                    if torch.max(boardState[:, coordinates[0] - x, coordinates[1] + x]):
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
                    if torch.max(boardState[:, coordinates[0] + x, coordinates[1] - x]):
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
                    if torch.max(boardState[:, coordinates[0] + x, coordinates[1] + x]):
                        break
        
        return check

    # visibility for testing
    def getChildren(self):
        return self.children

    def __str__(self):
        return self.boardState

with open(r'D:\Machine Learning\DeepLearningChessAI\small_val_set.db', 'rb') as file:
    val_set = pickle.load(file)
    print(val_set[1][0])
node = Node(val_set[1][0])
node.createChildren()
print(node.getChildren())



# thought: might want a history class to track boardstate metadata during game / tree navigation (ie. rightRookCastle and leftRookCastle boolean flags)