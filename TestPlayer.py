from unittest import TestCase
from unittest.mock import MagicMock
import unittest
import torch
from Node import Node
from Player import Player
import random


class TestPlayer(TestCase):

    def initialBoard(self):
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

    def setUp(self):
        board = self.initialBoard()
        self.node = Node(board)
        self.cnn = MagicMock(side_effect = [0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1,
                                            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.4, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                                            0.4, 
                                            0.5, 
                                            0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
                                            0.2,
                                            0.95])

        self.player = Player(self.node, self.cnn, breadth=2, depth=2)
        

    def test_minimax(self):
        move = self.player.minimax(self.node, self.player.depth, True, -1, 2)

        self.assertEqual(5, move[0], "expected index 5")
        self.assertEqual(0.4, move[1], "expected value of 0.4 (0.95 means alpha beta prune failed)")



if __name__ == '__main__':
    unittest.main()
