import torch
import Node
import CNN
import copy
import re
import pickle

class Player():
    def __init__(self, node, cnn, color, depth=1, breadth=1):
        self.tree = node
        self.cnn = cnn

        self.color = color
        self.turn = "White"
        self.opponentColorChannels = list()
        self.isMaximizer = False

        if self.color == "Black":
            self.opponentColorChannels = [0,1,2,3,4,5]
            self.isMaximizer = False
        else:
            self.opponentColorChannels = [6,7,8,9,10,11]
            self.isMaximizer = True

        self.depth = depth              # might change to "difficulty" param
        self.breadth = breadth

    # initializes play loop of making move and accepting opponent move
    def play(self):
        while self.isMate(self.tree) == False:
            if self.color == self.turn:
                self.myTurn(self.tree, self.depth, self.breadth)
                self.nextTurn()
            else:
                self.opponentTurn(self.tree)
                self.nextTurn()
        
        if self.tree.color == "White":
            print("Black wins by CheckMate")
        else:
            print("White wins by CheckMate")
                
    # accepts input from user and moves piece        
    def opponentTurn(self, tree):
        board = copy.deepcopy(tree.getBoard())
        tree.createChildren()
        legalMoves = tree.getChildren()

        print("Make your move: select a square and then input an action. Input \"done\" to submit move.")
        move = ""
        coordinates = []
        isTurn = True

        while isTurn:
            move = input()              # get user input
            if move == "done":
                # change turn
                if self.color == "White":
                    board[12:14, :, :] = 0
                else:
                    board[12:14, :, :] = 1
                print(Node.Node(board))

                for line in legalMoves:

                    if torch.equal(line.getBoard(), board):
                        self.tree = line                    # change head of tree to match opponent's move
                        isTurn = False
                        print("Move submitted")
                        break
                
                if isTurn:
                    print("submitted move was not legal")
                else:
                    break

            elif re.search(r'\d', move):          # if has number -> indicates square selection
                coordinates = []
                coordinates.append(8 - int(move[1]))
                coordinates.append(ord(move[0]) - 97)
                print("Selected: " + str(coordinates))

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
        index = self.minimax(tree, depth, self.isMaximizer)
        children = tree.getChildren()

        print(index)

        self.tree = children[index[0]]                      #minimax return index... orrr?****START HERE
        print(self.tree)

    # searches for optimally evaluated move - returns a tuple of (index, value)
    def minimax(self, node, depth, isMaximizer, nodeIndex=-1):                   # alpha beta pruning***
        # if node not created children yet (first traversal)
        if not node.getChildren():
            node.createChildren()

        # base condition:  specified depth
        if depth == 0 or self.isMate(node):
            return (nodeIndex, self.cnn(torch.unsqueeze(node.getBoard(), 0)))

        # base condition: stalemate
        if self.isStalemate(node):
            return (nodeIndex, [0.5])
        
        if isMaximizer:
            # get children values and store highest
            lines = self.getPredictions(node)

            x = 0
            value = (0, -1)
            # for # of lines specified in breadth
            while x < self.breadth and len(lines):
                x += 1

                # find and remove most promising line
                best = max(lines)
                index = lines.index(best)
                lines.remove(best)

                # store highest value & index of most promising lines
                proxy = self.minimax(node.getChildren()[index], depth-1, False, nodeIndex=index)
                if value[1] < proxy[1]:
                    value = proxy

        else:
            # get children values and store highest
            lines = self.getPredictions(node)

            x = 0
            value = (0, 2)
            # for # of lines specified in breadth
            while x < self.breadth and len(lines):
                x += 1

                # find and remove most promising line
                best = min(lines)
                index = lines.index(best)
                lines.remove(best)

                # store lowest value of most promising lines
                # store highest value & index of most promising lines
                proxy = self.minimax(node.getChildren()[index], depth-1, True, nodeIndex=index)
                if value[1] > proxy[1]:
                    value = proxy
        
        # return own index unless initializing method call
        if nodeIndex == -1:
            pass
        else:
            value = (nodeIndex, value[1])

        return value

        # be mindful of checkmate, stalemate, 3 fold repeat

    def getPredictions(self, node):             # assumes node already created children
        # get predictions on each child
        predictions = list()
        children = node.getChildren()
        for child in children:
            predictions.append(self.cnn(torch.unsqueeze(child.getBoard(), 0)))

        return predictions

    def isMate(self, node):                     # assumes node already created children
        isMate = False

        children = node.getChildren()

        if not children:                        # if leaf node & in check
            if node.inCheck(node.getBoard()):
                isMate = True

        return isMate

    def isStalemate(self, node):                # assumes node already created children
        isStalemate = False

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


def initialBoard():
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

board = initialBoard()
game = Node.Node(board)
network = 0
with open('D:\Machine Learning\DeepLearningChessAI\CNN_yankee2.cnn', 'rb') as file:
    network = pickle.load(file)
    #network.cuda()
player = Player(game, network, "White", 3, 3)
player.play()


### DEBUGGING
# list index out of range line 116 {done}}}

