import os
import subprocess
import torch
import numpy as np
from belavkin.belrl.tictactoe import TicTacToeGame
from belavkin.belrl.models import PolicyValueNet

def run_rl_experiment(optimizer):
    """Runs a single RL experiment."""
    print(f"Starting RL experiment for {optimizer}...")
    command = [
        'python3', 'belavkin/scripts/train_rl.py',
        '--game', 'tictactoe',
        '--optimizer', optimizer,
        '--num_episodes', '100',
        '--num_simulations', '25'
    ]

    # Run the command
    subprocess.run(command, env={**os.environ, 'PYTHONPATH': '.'})
    print(f"Finished RL experiment for {optimizer}.")

def pit(model1, model2, num_games):
    """Pits two models against each other."""
    game = TicTacToeGame()
    model1_wins = 0
    model2_wins = 0
    draws = 0

    for i in range(num_games):
        print(f"Pitting game {i+1}/{num_games}")
        game.board = np.zeros((3, 3), dtype=int)
        player = 1
        while not game.is_game_over(game.board):
            if player == 1:
                current_player_model = model1
            else:
                current_player_model = model2

            board_state = game.get_canonical_form(game.board, player)
            board_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Add batch and channel dims
            policy, _ = current_player_model(board_tensor)
            action = torch.argmax(policy).item()

            legal_moves = game.get_legal_moves(game.board)
            if not legal_moves:
                break
            if action not in legal_moves:
                action = np.random.choice(legal_moves)

            row, col = action // 3, action % 3
            if game.board[row, col] == 0:
                game.board[row, col] = player
            player *= -1

        result = game.get_game_result(game.board)
        if result == 1:
            model1_wins += 1
        elif result == -1:
            model2_wins += 1
        else:
            draws += 1

    return model1_wins, model2_wins, draws

def main():
    """Runs the full RL benchmark suite."""
    print("Starting RL benchmarks...")
    optimizers = ['belopt', 'adam']

    for optimizer in optimizers:
        run_rl_experiment(optimizer)

        # Save the model
        if os.path.exists('latest.pth'):
            model = PolicyValueNet(board_size=3, num_actions=TicTacToeGame().action_size, input_channels=1)
            model.load_state_dict(torch.load('latest.pth'))
            torch.save(model.state_dict(), f'belavkin/results/{optimizer}_model.pth')
        else:
            print(f"Could not find latest.pth for optimizer {optimizer}. Skipping...")

    # Evaluate the models
    if os.path.exists('belavkin/results/belopt_model.pth') and os.path.exists('belavkin/results/adam_model.pth'):
        print("Evaluating models...")
        belopt_model = PolicyValueNet(board_size=3, num_actions=TicTacToeGame().action_size, input_channels=1)
        belopt_model.load_state_dict(torch.load('belavkin/results/belopt_model.pth'))

        adam_model = PolicyValueNet(board_size=3, num_actions=TicTacToeGame().action_size, input_channels=1)
        adam_model.load_state_dict(torch.load('belavkin/results/adam_model.pth'))

        belopt_wins, adam_wins, draws = pit(belopt_model, adam_model, 100)
        print(f"BelOpt vs Adam: {belopt_wins} - {adam_wins} ({draws} draws)")

        adam_wins, belopt_wins, draws = pit(adam_model, belopt_model, 100)
        print(f"Adam vs BelOpt: {adam_wins} - {belopt_wins} ({draws} draws)")
    else:
        print("Could not find model files for evaluation. Skipping...")

    print("RL benchmarks complete.")

if __name__ == '__main__':
    main()
