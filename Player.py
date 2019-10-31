import torch
import Node
import CNN
import copy
import re

class Player():
    def __init__(self, node, cnn, color, depth=1, breadth=1):
        self.tree = node
        #self.boardState = node.getBoard()
        self.cnn = cnn

        self.color = color
        self.turn = "White"
        self.opponentColorChannels = list()
        if self.color == "Black":
            self.opponentColorChannels = [0,1,2,3,4,5]
        else:
            self.opponentColorChannels = [6,7,8,9,10,11]

        self.depth = depth              # might change to "difficulty" param
        self.breadth = breadth

    # initializes play loop of making move and accepting opponent move
    def play(self):
        while isMate(node) == False:
            if self.color == self.turn:
                self.myTurn(self.tree, self.depth, self.breadth)
                self.nextTurn()
            else:
                self.opponentTurn(self.tree)
                self.nextTurn()
                
    # accepts input from user and moves piece        
    def opponentTurn(self, tree):
        board = copy.deepCopy(tree.getBoard())
        tree.createChildren()
        legalMoves = tree.getChildren()

        print("Make your move: select a square and then input an action. Input \"done\" to submit move.)
        move = ""
        coordinates = []
        isTurn = True

        while isTurn:
            move = input()              # get user input
            if move == "done":
                for line in legalMoves:
                    if line.getBoard() == board:
                        tree = line                    # change head of tree to match opponent's move
                        isTurn = False
                        print("Move submitted")
                        break
                
                if isTurn:
                    print("submitted move was not legal")
                else:
                    break

            elif re.search(r'\d', move):          # if has number -> indicates square selection
                coordinates[0] = ord(move[0]) - 97
                coordinates[1] = 8 - int(move[1])
                print("Selected: " + coordinates)

            elif move == "clear":
                if coordinates:
                    board[0:12, coordinates[0], coordinates[1]] = 0
                    print(Node.Node(board))
                else:
                    print("first, select a square")

            elif len(move) == 1:
                if move == "K":
                    move = 0
                elif move == "Q":
                    move = 1
                elif move == "R":
                    move = 2
                elif move == "B":
                    move = 3
                elif move == "N":
                    move = 4
                elif move == "P":
                    move = 5
                else:
                    print("Piece not recognized")
                    continue

                if coordinates:
                    board[self.opponentColorChannels[move], coordinates[0], coordinates[1]] = 1
                    print(Node.Node(board))
                else:
                    print("first, select a square")
                
    def myTurn(self, tree, depth, breadth):
        index = self.minimax(tree, depth, breadth)
        children = tree.getChildren()
        tree = children[index]
        print(tree)
        
        # branch out the tree one layer
        # use cnn to evaluate each move
        #
        # select top __ prospective moves               breadth first miinimax
        # for each prospective move:
        # # branch out all opponent moves and evaluate
        # # store top __ prospOpp moves
        # # for each prospOpp:
        # # # etc...

    def minimax(self, tree, depth, breadth):
        return None
        # recursively? minimax through the tree breadth first (queue?) to specifed depth/breadth and return the best move
        # be mindful of checkmate, stalemate, 3 fold repeat

    def isMate(self, node):
        isMate = False

        node.createChildren()
        children = node.getChildren()

        if not children:                        # if leaf node & in check
            if node.inCheck():
                isMate = True

        return isMate

    def isStalemate(self):
        isStalemate = False

        node.createChildren()
        children = node.getChildren()

        if not children:                        # if leaf node & not in check
            if not node.inCheck():
                isStalemate = True

        return isStalemate

    def is3Fold(self):
        return None
        # check to see if draw by 3 fold                [*** not sure how this will implement, might take in node and list of indexes to check if repeating]

    def nextTurn(self):
        if self.turn == "White":
            self.turn = "Black"
        else:
            self.turn = "White"
