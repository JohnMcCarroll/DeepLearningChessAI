import ast
import inspect
import importlib
import os
import re
import sys
import textwrap
import types
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import copy


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def _import_self_play():
    return importlib.import_module("src.training.SelfPlayTraining")


class TestFileLocation(unittest.TestCase):

    def test_file_exists_in_src_training(self):
        path = os.path.join(PROJECT_ROOT, "src", "training", "SelfPlayTraining.py")
        self.assertTrue(os.path.isfile(path), "SelfPlayTraining.py must exist in src/training/")

    def test_module_importable(self):
        mod = _import_self_play()
        self.assertIsNotNone(mod)


class TestCNNModelCreation(unittest.TestCase):

    def test_creates_cnn_instance(self):
        from src.playing.CNN import CNN
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("CNN(", source, "Module should reference the CNN class")

    def test_new_cnn_passed_to_player(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        cnn_constructor_vars = set()
        player_call_args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    call_src = ast.dump(node.value)
                    if "CNN" in call_src:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                cnn_constructor_vars.add(target.id)
            if isinstance(node, ast.Call):
                call_src = ast.dump(node)
                if "Player" in call_src:
                    for arg in node.args + [kw.value for kw in node.keywords]:
                        if isinstance(arg, ast.Name):
                            player_call_args.append(arg.id)
                        elif isinstance(arg, ast.Attribute):
                            player_call_args.append(ast.dump(arg))
        passed = any(var in player_call_args for var in cnn_constructor_vars)
        self.assertTrue(
            passed,
            "A newly instantiated CNN should be passed to the Player class at the start of training",
        )

class TestPlayerUsage(unittest.TestCase):

    def test_references_player_class(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("Player", source, "Module should use the Player class")

    def test_player_depth_and_breadth(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("depth=5", source, "Depth of 5 should appear in source")
        self.assertIn("breadth=4", source, "Breadth of 4 should appear in source")


class TestSelfPlayLoopStructure(unittest.TestCase):

    def test_loop_iterations_count(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("10", source, "Self play loop should iterate 10 times")

    def test_100_games_per_iteration(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("100", source, "Should play 100 games per iteration")

    def test_uses_multiprocessing(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_mp = "multiprocessing" in source or "Pool" in source or "Process" in source
        self.assertTrue(has_mp, "Should use multiprocessing for parallel game play")


class TestPlayCoordinationFunction(unittest.TestCase):

    def test_has_play_coordination_function(self):
        mod = _import_self_play()
        functions = [
            name for name, obj in inspect.getmembers(mod)
            if isinstance(obj, types.FunctionType)
        ]
        self.assertTrue(
            len(functions) >= 1,
            "Module should define at least one function for coordinating play between two Players",
        )

    def test_coordination_function_involves_two_players(self):
        mod = _import_self_play()
        funcs = [
            obj for name, obj in inspect.getmembers(mod)
            if isinstance(obj, types.FunctionType)
        ]
        found = False
        for func in funcs:
            src = inspect.getsource(func)
            if "Player" in src or "player" in src.lower():
                found = True
                break
        self.assertTrue(found, "A function should coordinate play between Player instances")


class TestRandomColorAssignment(unittest.TestCase):

    def test_uses_random_for_color(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_random = "random" in source.lower()
        self.assertTrue(has_random, "Should randomly assign black/white pieces at game start")


class TestGameResultValues(unittest.TestCase):

    def test_result_values_in_source(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("1.0", source, "White win result (1.0) should appear")
        self.assertIn("0.0", source, "White loss result (0.0) should appear")
        self.assertIn("0.5", source, "Draw result (0.5) should appear")


class TestDataDictionary(unittest.TestCase):

    def test_uses_dict_for_data(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_dict = "dict" in source or "{}" in source
        self.assertTrue(has_dict, "Should use a dictionary to store board state data")

    def test_string_keys_for_board_states(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_str_conversion = "str(" in source or "repr(" in source or "boardToString" in source or "str_board" in source
        self.assertTrue(
            has_str_conversion,
            "Dictionary keys should be string representations of board state tensors",
        )

    def test_dict_values_are_tuples_with_avg_and_count(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_tuple = "tuple" in source or re.search(r'\([^)]*,[^)]*\)', source) is not None
        self.assertTrue(
            has_tuple,
            "Dictionary values should be tuples of (average_result, count)",
        )


class TestOpponentUpdate(unittest.TestCase):

    def test_first_opponent_is_betazero(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("BetaZero.cnn", source, "First opponent should be loaded from BetaZero.cnn")

    def test_opponent_updated_to_frozen_copy(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_freeze = (
            "deepcopy" in source
            or "copy" in source
            or "clone" in source
            or "state_dict" in source
            or "load_state_dict" in source
            or "requires_grad" in source
            or "no_grad" in source
        )
        self.assertTrue(
            has_freeze,
            "Opponent should be updated to a frozen copy of the CNN's old weights",
        )


class TestModelTraining(unittest.TestCase):

    def test_uses_mse_loss(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_mse = "mse_loss" in source or "MSELoss" in source
        self.assertTrue(has_mse, "Should use mean squared error loss for training")

    def test_uses_adam_optimizer(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("Adam", source, "Should use Adam optimizer (from TrainingLoop.py)")

    def test_learning_rate_from_training_loop(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("0.0001", source, "Learning rate should be 0.0001 based on TrainingLoop.py")

    def test_epochs_increment_each_iteration(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_increment = "+=" in source or "+ 1" in source or "+1" in source
        self.assertTrue(
            has_increment,
            "Number of training epochs should increment with each self play loop iteration",
        )

    # def test_batch_size_from_training_loop(self):
    #     mod = _import_self_play()
    #     source = inspect.getsource(mod)
    #     self.assertIn("50", source, "Batch size should be 50 based on TrainingLoop.py")


class TestMoveLimit(unittest.TestCase):

    def test_200_move_limit(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("200", source, "Games exceeding 200 moves should be terminated as a draw")


class TestModelSaving(unittest.TestCase):

    def test_saves_to_self_play_models_dir(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn(
            "self_play_models",
            source,
            "Models should be saved to the self_play_models directory",
        )

    def test_creates_output_directory(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_mkdir = "makedirs" in source or "mkdir" in source or "os.path.exists" in source or "Path" in source
        self.assertTrue(
            has_mkdir,
            "Should create the self_play_models directory if it doesn't exist",
        )

    def test_uses_torch_save(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("torch.save", source, "Should use torch.save to save models")


class TestEpochStartsAtOne(unittest.TestCase):

    def test_initial_epoch_is_one(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        found_n_eq_1 = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                        if node.value.value == 1:
                            found_n_eq_1 = True
                            break
        self.assertTrue(
            found_n_eq_1,
            "Epoch count n should start at 1",
        )


class TestSelfPlayModelsDirectory(unittest.TestCase):

    def test_directory_at_project_root(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertNotIn(
            "src/self_play_models",
            source,
            "self_play_models should be at the project root, not inside src/",
        )


class TestGamePlayFunction(unittest.TestCase):

    def test_play_game_function_returns_board_states_and_result(self):
        mod = _import_self_play()
        funcs = {
            name: obj
            for name, obj in inspect.getmembers(mod)
            if isinstance(obj, types.FunctionType)
        }
        self.assertTrue(
            len(funcs) >= 1,
            "Should define at least one function for playing a game",
        )
        for name, func in funcs.items():
            src = inspect.getsource(func)
            if "result" in src.lower() or "board" in src.lower() or "game" in src.lower():
                has_return = "return" in src
                self.assertTrue(
                    has_return,
                    f"Function '{name}' that handles game play should return data",
                )
                break


class TestFreezeOpponentWeights(unittest.TestCase):

    def test_opponent_weights_frozen(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        has_freeze = (
            "requires_grad" in source
            or "no_grad" in source
            or "freeze" in source.lower()
            or "eval()" in source
            or ".param" in source
        )
        self.assertTrue(
            has_freeze,
            "Opponent model weights should be frozen (not trainable)",
        )


class TestLoopExitsAfter10Iterations(unittest.TestCase):

    def test_loop_bound(self):
        mod = _import_self_play()
        source = inspect.getsource(mod)
        self.assertIn("10", source, "Self play loop should exit after 10 iterations")


if __name__ == "__main__":
    unittest.main()
