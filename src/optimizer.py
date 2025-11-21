import torch
from torch.optim.optimizer import Optimizer
import math

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, adaptive_decay=True, panic_threshold=1e-5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        adaptive_decay=adaptive_decay, panic_threshold=panic_threshold)
        super(BelavkinOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Returns: (loss, innovation_norm, is_panicking)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_innovation_sq = 0.0
        total_param_norm = 0.0

        # First pass: Compute Global Innovation (GNS proxy)
        # We accumulate across all groups to get a global metric
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format) # m_t
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format) # v_t

                m_t = state['exp_avg']

                # Innovation = Gradient - Expected Gradient (Momentum)
                innovation = grad - m_t

                # We use .norm() here. Note: .item() causes CPU-GPU sync,
                # but is necessary to get the scalar for the Python logic below.
                # Optimizing this for pure GPU execution would require moving logic to CUDA kernels.
                total_innovation_sq += innovation.norm().item() ** 2
                total_param_norm += p.norm().item() ** 2

        # Calculate Global Metrics
        gns_proxy = total_innovation_sq / (total_param_norm + 1e-12)

        # Second pass: Update parameters
        # We assume panic_threshold is consistent or we take the one from the first group for global decision
        # Ideally, panic logic should be per-group or clearly defined as global.
        # For this implementation, we use the first group's threshold if available, or default.
        panic_threshold = self.param_groups[0].get('panic_threshold', 1e-5)
        is_panicking = gns_proxy < panic_threshold

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            # Dynamic Decay Rule (AdamW-B Logic)
            if group['adaptive_decay']:
                # If innovation is high, increase decay to prevent Softmax Collapse
                # If innovation is low (stuck), reduce decay to allow drift
                adaptive_wd = weight_decay * (1 + gns_proxy)
            else:
                adaptive_wd = weight_decay

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                state['step'] += 1

                m_t = state['exp_avg']
                v_t = state['exp_avg_var']

                # Update Belief (Momentum)
                m_t.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update Variance (AdaBelief style: variance of innovation)
                innovation = grad - m_t
                v_t.mul_(beta2).addcmul_(innovation, innovation, value=1 - beta2)

                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # AdamW-B Update
                denom = (v_t.sqrt() / math.sqrt(bias_correction2)) + group['eps']
                step_size = group['lr'] / bias_correction1

                # Decoupled Weight Decay (Dynamic)
                p.mul_(1 - step_size * adaptive_wd)

                # Gradient Step
                p.addcdiv_(m_t, denom, value=-step_size)

        return loss, gns_proxy, is_panicking
