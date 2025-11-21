import torch
from torch.optim.optimizer import Optimizer

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
        for group in self.param_groups:
            beta1, beta2 = group['betas']

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
                # Note: This is the "Surprise"
                innovation = grad - m_t
                total_innovation_sq += innovation.norm().item() ** 2
                total_param_norm += p.norm().item() ** 2

        # Calculate Global Metrics
        gns_proxy = total_innovation_sq / (total_param_norm + 1e-12)
        # Assuming one group or using the last group's threshold.
        # In benchmarking we typically have consistent settings.
        # We grab the first group's threshold for global decision.
        panic_threshold = self.param_groups[0]['panic_threshold']
        is_panicking = gns_proxy < panic_threshold

        # Second pass: Update parameters
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
                # Use scalar math for bias correction sqrt to avoid tensor overhead inside loop
                bias_correction2_sqrt = bias_correction2 ** 0.5
                denom = (v_t.sqrt() / bias_correction2_sqrt) + group['eps']
                step_size = group['lr'] / bias_correction1

                # Decoupled Weight Decay (Dynamic)
                p.mul_(1 - step_size * adaptive_wd)

                # Gradient Step
                p.addcdiv_(m_t, denom, value=-step_size)

        return loss, gns_proxy, is_panicking
