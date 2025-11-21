
import torch
from torch.optim import Optimizer
import math

class AdamWB(Optimizer):
    """
    AdamW-B: AdamW with Belavkin-inspired Innovation-Gated Regularization and Damping.

    Core Innovations:
    1. Innovation-Based Damping: Scales learning rate by exp(-gamma * normalized_innovation).
       If innovation (gradient surprise) is high, we "tap the brakes" (damping).

    2. Collapse Force (Heuristic Regularization): Adds a restoring force to the update
       proportional to (innovation^2 - variance_of_innovation).
       This acts as a first-order approximation of minimizing temporal inconsistency.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
            (default: False)
        gamma_damping (float, optional): coefficient for innovation-based damping (default: 1e-2)
        lambda_collapse (float, optional): coefficient for the collapse force (default: 0.0)
        beta_innovation (float, optional): smoothing factor for innovation variance tracking (default: 0.999)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False,
                 gamma_damping=0.0, lambda_collapse=0.0, beta_innovation=0.999,
                 signal_mode='innovation'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        gamma_damping=gamma_damping, lambda_collapse=lambda_collapse,
                        beta_innovation=beta_innovation, signal_mode=signal_mode)
        super(AdamWB, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWB, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            # Belavkin specific states
            innovation_vars = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamWB does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        # Belavkin: Innovation Variance Tracking (v_innov)
                        state['exp_avg_innov_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    innovation_vars.append(state['exp_avg_innov_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state['step'] += 1
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            beta_innov = group['beta_innovation']
            gamma_damp = group['gamma_damping']
            lambda_col = group['lambda_collapse']
            mode = group.get('signal_mode', 'innovation')

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                v_innov = innovation_vars[i]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Perform AdamW weight decay
                if group['weight_decay'] != 0:
                    param.mul_(1 - group['lr'] * group['weight_decay'])

                # 1. Innovation Calculation: delta = grad - m_t (approx)
                # Note: exp_avg is m_{t-1} effectively before update, or we can use current m_t.
                # The prompt says "g_t - m_t" where m_t is AdamW's first moment.
                # Let's update moments first, then calculate innovation?
                # Or calculate innovation against OLD moment?
                # Kalman Innovation is (measurement - prediction). Prediction is based on OLD state.
                # So Innovation = grad - exp_avg (before update).

                # Capture innovation using previous moment
                if mode == 'innovation':
                    innovation = grad - exp_avg
                    innovation_sq = innovation.pow(2)
                elif mode == 'random':
                    # Random noise as signal
                    innovation = torch.randn_like(grad)
                    innovation_sq = innovation.pow(2)
                elif mode == 'magnitude':
                    # Gradient magnitude as signal
                    innovation = grad
                    innovation_sq = grad.pow(2)
                else:
                    raise ValueError(f"Unknown signal mode: {mode}")

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # AMSGrad logic
                if group['amsgrad']:
                    torch.max(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                    denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Update Innovation Variance
                # v_{t} = beta * v_{t-1} + (1-beta) * innovation^2
                v_innov.mul_(beta_innov).add_(innovation_sq, alpha=1 - beta_innov)

                # Bias correct innovation variance for stable scaling
                bias_correction_innov = 1 - beta_innov ** step
                v_innov_corrected = v_innov / bias_correction_innov

                # --- Belavkin Heuristics ---

                # 1. Damping Factor
                # effective_lr = lr * exp(-gamma * innovation^2 / (v_innov + eps))
                # We use element-wise damping.
                if gamma_damp > 0:
                     damping_term = torch.exp(-gamma_damp * innovation_sq / (v_innov_corrected + group['eps']))
                else:
                     damping_term = 1.0

                step_size = group['lr'] * damping_term / bias_correction1

                # 2. Collapse Force
                # force = lambda * (innovation^2 - v_innov)
                # We add this to the update as a modification of the gradient term.

                if lambda_col != 0:
                    collapse_force = lambda_col * (innovation_sq - v_innov_corrected)
                else:
                    collapse_force = 0.0

                # Final Update
                # We manually compute the update to integrate damping and collapse force

                numerator = exp_avg
                if isinstance(collapse_force, torch.Tensor) or collapse_force != 0:
                    numerator = numerator + collapse_force

                # Calculate the update step
                # step_size is a tensor (damping_term is tensor).
                step = step_size * (numerator / denom)

                # Apply update
                param.data.add_(-step)

        return loss
