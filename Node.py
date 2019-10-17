import torch
import pickle


class Node:
    def __init__(self, boardState):
        self.boardState = boardState            # the position of the pieces
        self.color = ""                         # the color to move
        self.colorChannels
        self.children = set()                   # the set of all possible board states after next move

    def createChildren(self):

        # determine color 

        colorChannels = []

        if self.boardState[13, 0, 0] == 0:
            self.color = "White"
            self.colorChannels = [0, 1, 2, 3, 4, 5]
        else:
            self.color = "Black"
            self.colorChannels = [6, 7, 8, 9, 10, 11]
        
        # iterate through each square on board seeing if piece of color resides there

        for channel in colorChannels:
            # find piece locations
            locations = torch.nonzero(self.boardState[channel, :, :])

            for coordinates in locations:

                print(coordinates)

                # identify piece and call helper method to generate all legal moves

                if channel % 6 == 0:
                    moves = kingMoves(boardState, colorChannels, coordinates)
                elif channel % 6 == 1:
                    #queenMoves(boardState, channel, coordinates)
                elif channel % 6 == 2:
                    #rookMoves(boardState, channel, coordinates)
                elif channel % 6 == 3:
                    #bishopMoves(boardState, channel, coordinates)
                elif channel % 6 == 4:
                    #knightMoves(boardState, channel, coordinates)
                elif channel % 6 == 5:
                    #pawnMoves(boardState, channel, coordinates)
                else:
                    print(channel)


        # create a new node to hold boardState

        # add each node to children set...
    
    def kingMoves(self, boardState, colorChannels, coordinates):
        moves = list()
        originalBoard = boardState

        # piece movement
        for row in [-1, 0, 1]:
            for col in [-1, 0, 1]:
                if row = 0 and col = 0:
                    continue    # skip if checking same spot
                else:
                    # check to ensure own color piece not in way
                    if boardState[colorChannels, coordinates[0] + row, coordinates[1] + col] = 0:
                        # clear Channel
                        board[colorChannels[0], :, :] = 0
                        # remove captured piece
                        board[0:12, coordinates[0] + row, coordinates[1] + col] = 0
                        # place king
                        board[colorChannels[0], coordinates[0] + row, coordinates[1] + col] = 1

                        # make sure legal move
                        if not inCheck(boardState):
                            # add possible move to list
                            moves.append(boardState)

                    # refresh boardState
                    boardState = originalBoard

        # castling
        rookLocs = torch.nonzeros(boardState[colorChannels[2], :, :])

        if color = "White":
            if coordinates = [7,4] and rookLocs.count([7,0]):       
                if #check to see if interim squares empty, if yes move pieces

            if coordinates = [7,4] and rookLocs.count([7,7]):


        return moves

    def inCheck(self, boardState)
        # need a list of squares opponent is attacking...

with open(r'D:\Machine Learning\DeepLearningChessAI\small_val_set.db', 'rb') as file:
    val_set = pickle.load(file)
    print(val_set[1][0])
node = Node(val_set[1][0])
node.createChildren()
