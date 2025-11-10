import argparse
import torch
import torch.optim as optim
from belavkin.belrl.game import ChessGame
from belavkin.belrl.tictactoe import TicTacToeGame
from belavkin.belrl.models import PolicyValueNet
from belavkin.belrl.alphazero_loop import AlphaZero, ReplayBuffer
from belavkin.belopt.optim import BelOpt

def main(args):
    # Set up the game
    if args.game == 'chess':
        game = ChessGame()
        model = PolicyValueNet(board_size=8, num_actions=game.action_size, input_channels=6)
    elif args.game == 'tictactoe':
        game = TicTacToeGame()
        model = PolicyValueNet(board_size=3, num_actions=game.action_size, input_channels=1)
    else:
        raise ValueError(f"Unknown game: {args.game}")

    # Set up the optimizer
    if args.optimizer == 'belopt':
        optimizer = BelOpt(model.parameters(), lr=args.lr, gamma0=args.gamma0, beta0=args.beta0)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Set up the replay buffer
    replay_buffer = ReplayBuffer(capacity=args.replay_buffer_capacity)

    # Set up the AlphaZero agent
    alphazero = AlphaZero(game, model, optimizer, replay_buffer, args)

    # Run the training
    alphazero.run()

    # Save the model
    torch.save(model.state_dict(), 'latest.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Belavkin RL Training')
    parser.add_argument('--game', type=str, default='tictactoe', choices=['chess', 'tictactoe'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['belopt', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma0', type=float, default=0.0)
    parser.add_argument('--beta0', type=float, default=0.0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_simulations', type=int, default=25)
    parser.add_argument('--replay_buffer_capacity', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_train_steps', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--cpuct', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
