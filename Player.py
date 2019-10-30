import torch
import Node
import CNN

class Player():
    def __init__(self, node, cnn, color, depth=1, breadth=1):
        self.tree = node
        # self.
        self.cnn = cnn
        self.color = color
        self.depth = depth              # might change to "difficulty" param
        self.breadth = breadth

    def play(self):
        return None
        # initializes play loop of making move and accepting opponent move

    def opponentTurn(self):
        return None
        # accepts console input and moves piece for user

    def myTurn(self):
        return None
        # branch out the tree one layer
        # use cnn to evaluate each move
        #
        # select top __ prospective moves               breadth first miinimax
        # for each prospective move:
        # # branch out all opponent moves and evaluate
        # # store top __ prospOpp moves
        # # for each prospOpp:
        # # # etc...

    def minimax(self):
        return None
        # recursively? minimax through the tree breadth first (queue?) to specifed depth/breadth and return the best move
        # be mindful of checkmate, stalemate, 3 fold repeat

    def isMate(self):
        return None
        # check board to see if mate

    def isStalemate(self):
        return None
        # check board to see if stalemate...            [*** might be useless -> stalemate would be leaf node w/ boardstate not inCheck]

    def is3Fold(self):
        return None
        # check to see if draw by 3 fold                [*** not sure how this will implement, might take in node and list of indexes to check if repeating]
