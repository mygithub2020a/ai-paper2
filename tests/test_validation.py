"""
Quick validation tests for the Belavkin framework.

This script performs basic smoke tests to ensure everything is working.
"""

import sys
import traceback

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")

        from track1_optimizer import BelavkinOptimizer, BelavkinOptimizerSGLD, BelavkinOptimizerMinimal
        print("  ✓ Track 1 optimizer imports")

        from track2_rl import DensityMatrix, BelavkinRLAgent, BelavkinRLTrainer
        print("  ✓ Track 2 RL imports")

        from experiments import synthetic_tasks, benchmark
        print("  ✓ Experiment modules")

        from utils import visualization
        print("  ✓ Utility modules")

        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        traceback.print_exc()
        return False


def test_optimizer_basic():
    """Test basic optimizer functionality."""
    print("\nTesting BelavkinOptimizer basic functionality...")
    try:
        import torch
        import torch.nn as nn
        from track1_optimizer import BelavkinOptimizer

        # Create simple model
        model = nn.Linear(10, 2)
        optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3, gamma=1e-4, beta=1e-2)

        # Single forward-backward pass
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))

        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()

        print("  ✓ Optimizer step successful")
        return True
    except Exception as e:
        print(f"  ✗ Optimizer test failed: {e}")
        traceback.print_exc()
        return False


def test_rl_basic():
    """Test basic RL functionality."""
    print("\nTesting BelavkinRL basic functionality...")
    try:
        import torch
        from track2_rl import DensityMatrix, BelavkinRLAgent

        # Test density matrix
        rho = DensityMatrix(state_dim=4, rank=3)
        matrix = rho.as_matrix()
        entropy = rho.entropy()
        print(f"  ✓ DensityMatrix created (entropy: {entropy:.4f})")

        # Test agent
        agent = BelavkinRLAgent(state_dim=4, action_dim=2, rank=5)
        action = agent.select_action(training=False)
        print(f"  ✓ Agent action selection (action: {action})")

        # Test belief update
        obs = torch.randn(4)
        agent.update_belief(action=0, observation=obs, reward=1.0)
        print("  ✓ Belief update successful")

        return True
    except Exception as e:
        print(f"  ✗ RL test failed: {e}")
        traceback.print_exc()
        return False


def test_synthetic_tasks():
    """Test synthetic task creation."""
    print("\nTesting synthetic task creation...")
    try:
        from experiments.synthetic_tasks import create_modular_task, create_sparse_parity_task

        # Test modular task
        model, train_loader, test_loader = create_modular_task(
            p=11, operation='add', hidden_dim=32, n_layers=1, batch_size=16
        )
        print(f"  ✓ Modular task created ({len(train_loader)} train batches)")

        # Test sparse parity task
        model, train_loader, test_loader = create_sparse_parity_task(
            n_bits=5, k_sparse=2, hidden_dims=[32], n_samples=100, batch_size=16
        )
        print(f"  ✓ Sparse parity task created ({len(train_loader)} train batches)")

        return True
    except Exception as e:
        print(f"  ✗ Synthetic task test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*60)
    print("BELAVKIN FRAMEWORK VALIDATION TESTS")
    print("="*60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Optimizer Basic", test_optimizer_basic()))
    results.append(("RL Basic", test_rl_basic()))
    results.append(("Synthetic Tasks", test_synthetic_tasks()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("="*60))
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("Ready to run experiments!")
    else:
        print("SOME TESTS FAILED ✗")
        print("Please fix errors before running experiments.")
    print("="*60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
