import torch

def optimizer_to(optimizer, device):
    """
    Move the state of a PyTorch optimizer to a specified device.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to move.
        device (torch.device or str): The target device.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
