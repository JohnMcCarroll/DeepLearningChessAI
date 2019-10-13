import torch


class Node:
    def __init__(self, boardState):
        self.boardState = boardState            # the position of the pieces
        self.color = ""                         # the color to move
        self.children = set()                   # the set of all possible board states after next move

    def createChildren():

        # determine color (might be better in constructor?)

        colorChannels = []

        if self.boardState[13][0][0] == 0:
            self.color = "White"
            colorChannels = [0:5]   #?
        else:
            self.color = "Black"
            colorChannels = [6:11]  #?
        
        # iterate through each square on board seeing if piece of color resides there

        for channel in colorChannels:
            #dummy, locationByRow = torch.max(self.boardState[channel][:][:], 1)     ##ASSUMING ONLY ONE PIECE PER ROW...***
            location = torch.nonzero(self.boardState[channel][:][:])

            if channel % 6 == 0:
                kingMoves(boardState, channel)
            elif channel % 6 == 1:
                queenMoves(boardState, channel)
            elif channel % 6 == 2:
                rookMoves(boardState, channel)
            elif channel % 6 == 3:
                bishopMoves(boardState, channel)
            elif channel % 6 == 4:
                knightMoves(boardState, channel)
            elif channel % 6 == 5:
                pawnMoves(boardState, channel)
            else:
                print(channel)

        # when piece identified, generate a board state corresponding to each legal move

        # create a new node to hold boardState

        # add each node to children set..