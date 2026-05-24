import unittest
import torch
import time
from src.playing.Node import Node
from src.playing.Player import Player


class TestPlayerSpeed(unittest.TestCase):
    
    def initialBoard(self):
        board = torch.zeros([14, 8, 8])
        board[0, 7, 4] = 1
        board[1, 7, 3] = 1
        board[2, 7, 0] = 1
        board[2, 7, 7] = 1
        board[3, 7, 2] = 1
        board[3, 7, 5] = 1
        board[4, 7, 1] = 1
        board[4, 7, 6] = 1
        board[5, 6, :] = 1
        board[6, 0, 4] = 1
        board[7, 0, 3] = 1
        board[8, 0, 0] = 1
        board[8, 0, 7] = 1
        board[9, 0, 2] = 1
        board[9, 0, 5] = 1
        board[10, 0, 1] = 1
        board[10, 0, 6] = 1
        board[11, 1, :] = 1
        return board
    
    def test_player_speed(self):
        board = self.initialBoard()
        node = Node(board)
        
        network = torch.load('BetaZero.cnn', weights_only=False, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = network.to(device)
        
        player = Player(node, network, "White", depth=1, breadth=1)
        
        depth = 1
        breadth = 1
        increment_depth = True
        
        print("\n=== Player Speed Test ===")
        
        while True:
            player.depth = depth
            player.breadth = breadth
            
            board = self.initialBoard()
            test_node = Node(board)
            
            start_time = time.time()
            player.myTurn(test_node, depth, breadth)
            elapsed_time = time.time() - start_time
            
            print(f"Depth: {depth}, Breadth: {breadth}, Runtime: {elapsed_time:.4f} seconds")
            
            if elapsed_time > 30:
                print(f"\nTest ended: Runtime exceeded 30 seconds")
                break
            
            if increment_depth:
                depth += 1
            else:
                breadth += 1
            
            increment_depth = not increment_depth
        
        self.assertGreaterEqual(depth, 3, "Depth should be at least 3")
        self.assertGreaterEqual(breadth, 3, "Breadth should be at least 3")
        print(f"\nTest passed: Final depth={depth}, breadth={breadth}")


if __name__ == '__main__':
    unittest.main()
