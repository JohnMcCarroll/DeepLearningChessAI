import torch
import Node


def promotion():
    # PAWN PROMOTION TEST
    board = torch.zeros([14, 8, 8])
    board[5, 1, 0] = 1
    board[0, 7, 0] = 1

    node = Node.Node(board)
    node.WKC = False
    node.WQC = False

    print("PARENT:")
    print(node)

    print("CHILDREN:")
    node.createChildren()
    children = node.getChildren()
    for child in children:
        print(child)

def enPassant():
    board = torch.zeros([14, 8, 8])
    board[5, 6, 0] = 1
    board[11, 4, 1] = 1

    board[0, 7, 7] = 1
    board[6, 5, 7] = 1

    print("initial:")
    node = Node.Node(board)
    node.WKC = False
    node.WQC = False
    node.BKC = False
    node.BQC = False
    node.createChildren()

    children = node.getChildren()
    print(node)

    line = 0
    for child in children:
        line += 1
        print("CHILD " + str(line))
        print(child)
        child.createChildren()
        more = child.getChildren()

        print("GRANDKIDS...")
        for child2 in more:
            print(child2)

def castling():
    board = torch.zeros([14, 8, 8])
    board[6, 0, 4] = 1
    board[0, 7, 4] = 1
    board[8, 0, 0] = 1
    board[8, 0, 7] = 1
    board[2, 7, 3] = 1
    board[2, 7, 5] = 1
    board[12:14, :, :] = 1

    node = Node.Node(board)
    print("parent:")
    print(node)

    node.createChildren()
    children = node.getChildren()

    for child in children:
        print(child)

def queen():
    board = torch.zeros([14, 8, 8])
    board[6, 0, 0] = 1
    board[0, 7, 4] = 1
    board[7, 3, 4] = 1
    board[12:14, :, :] = 1
    board[1, 6, 4] = 1

    print("parent")
    node = Node.Node(board)
    print(node)

    node.createChildren()
    children = node.getChildren()
    for child in children:
        print(child)

def capturePromo():
    board = torch.zeros([14, 8, 8])
    board[11, 6, 7] = 1
    board[4, 7, 6] = 1
    board[6, 0, 0] = 1
    board[12:14, :, :] = 1

    node = Node.Node(board)
    node.WKC = False
    node.WQC = False
    node.BKC = False
    node.BQC = False

    print("PARENT:")
    print(node)

    print("CHILDREN:")
    node.createChildren()
    children = node.getChildren()
    for child in children:
        print(child)

capturePromo()

# BUGS:
#
# spontaneous castling (w/ rook generation)                 [testing constraint]
# pawn promote to knight -> pawn does not disapear          [fixed]
# no en passant                                             [fixed]
# duplicate pawn caps? (not sure if real)                   [not duplicated]
# queen collision detection on horizontal failure           [fixed]
# no capture promotions                                     [fixed]