import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from multiprocessing import Pool, cpu_count
import src.playing.CNN as CNN
import src.playing.Player as Player
import src.playing.Node as Node
import src.data.DataAlteration as DataAlteration
from pathlib import Path
from datetime import datetime


def initialBoard():
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


def play_game(model_state_dict, opponent_state_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_cnn = CNN.CNN()
    model_cnn.load_state_dict(model_state_dict)
    model_cnn.to(device)
    model_cnn.eval()
    
    opponent_cnn = CNN.CNN()
    opponent_cnn.load_state_dict(opponent_state_dict)
    opponent_cnn.to(device)
    opponent_cnn.eval()
    
    white_color = random.choice(["White", "Black"])
    black_color = "Black" if white_color == "White" else "White"
    
    if white_color == "White":
        white_cnn = model_cnn
        black_cnn = opponent_cnn
    else:
        white_cnn = opponent_cnn
        black_cnn = model_cnn
    
    board = initialBoard()
    node = Node.Node(board)
    
    white_player = Player.Player(node, white_cnn, "White", depth=4, breadth=4)
    black_player = Player.Player(node, black_cnn, "Black", depth=4, breadth=4)
    
    game_states = []
    current_player = white_player
    other_player = black_player
    
    max_moves = 200
    move_count = 0
    
    while move_count < max_moves:
        current_node = current_player.tree
        
        if not current_node.getChildren():
            current_node.createChildren()
        
        if current_player.isMate(current_node):
            if current_node.color == "White":
                result = 0.0
            else:
                result = 1.0
            break
        
        if current_player.isStalemate(current_node):
            result = 0.5
            break
        
        game_states.append(DataAlteration.boardToString(current_node.getBoard()))
        
        move_info = current_player.minimax(current_node, current_player.depth, current_player.isMaximizer, -1, 2)
        move_index = move_info[0]
        
        children = current_node.getChildren()
        next_node = children[move_index]
        
        current_player.tree = next_node
        other_player.tree = next_node
        
        current_player, other_player = other_player, current_player
        move_count += 1
    else:
        result = 0.5
    
    return game_states, result


def generate_self_play_data(model_state_dict, opponent_state_dict, num_games=100):
    num_processes = min(cpu_count(), num_games)
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(play_game, [(model_state_dict, opponent_state_dict)] * num_games)
    
    return results


def update_training_data(data_dict, game_results):
    for game_states, result in game_results:
        for state_string in game_states:
            if state_string in data_dict:
                avg_result, count = data_dict[state_string]
                new_count = count + 1
                new_avg = (avg_result * count + result) / new_count
                data_dict[state_string] = (new_avg, new_count)
            else:
                data_dict[state_string] = (result, 1)
    
    return data_dict


def train_model(model, data_dict, num_epochs, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    board_states = []
    targets = []
    
    for state_string, (avg_result, count) in data_dict.items():
        board_tensor = DataAlteration.stringToBoard(state_string)
        board_states.append(board_tensor)
        targets.append(avg_result)
    
    if len(board_states) == 0:
        return
    
    board_states = torch.stack(board_states).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        predictions = model(board_states)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")


def self_play_training_loop(num_iterations=10, games_per_iteration=100, output_dir=Path.cwd()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    opponent = torch.load('BetaZero.cnn', weights_only=False, map_location=device)
    opponent.eval()
    
    model = CNN.CNN()   # new model to train
    
    data_dict = {}
    
    for iteration in range(num_iterations):
        print(f"\n=== Self-Play Training Iteration {iteration + 1}/{num_iterations} ===")
        
        print(f"Generating data from {games_per_iteration} games...")
        model_state_dict = model.state_dict()
        opponent_state_dict = opponent.state_dict()
        
        game_results = generate_self_play_data(model_state_dict, opponent_state_dict, games_per_iteration)
        
        print(f"Updating training data dictionary...")
        data_dict = update_training_data(data_dict, game_results)
        print(f"Training data now contains {len(data_dict)} unique positions")
        
        print(f"Updating opponent model...")
        opponent = copy.deepcopy(model)
        opponent.eval()
        
        num_epochs = min(iteration + 1, 10)
        print(f"Training model for {num_epochs} epochs...")
        train_model(model, data_dict, num_epochs)
        
        print(f"Saving model...")
        output_file_path = output_dir / Path(f'BetaZero{iteration + 1}.cnn')
        torch.save(model, output_file_path)
        print(f"Iteration {iteration + 1} complete!")
    
    print("\n=== Self-Play Training Complete ===")


if __name__ == '__main__':
    # Create output directory
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%S")
    repo_root_path = Path(__file__).parent.parent.parent
    output_dir_path = repo_root_path / Path(f"self_play_models/{date_str}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    # Launch training
    self_play_training_loop(num_iterations=100, games_per_iteration=10, output_dir=output_dir_path)
