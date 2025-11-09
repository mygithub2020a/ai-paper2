import torch
from collections import deque

class BelOptim:
    def __init__(self, params, lr=0.01, gamma=0.1, alpha=0.1, beta1=0.9, beta2=0.999):
        """
        params: Model parameters
        lr: Base learning rate (η)
        gamma: Measurement strength (γ)
        alpha: Hamiltonian scaling
        beta1, beta2: Momentum factors (borrowed from Adam for stability)
        """
        self.params = list(params)
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

        # State tracking
        self.m = [torch.zeros_like(p) for p in self.params]  # First moment (filtered gradient)
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment (uncertainty)
        self.innovation = [torch.zeros_like(p) for p in self.params]  # Innovation process
        self.t = 0

    def step(self, gradients):
        self.t += 1

        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            # Compute innovation: dW̃ = dW - 2Re⟨ψ|L|ψ⟩dt
            # In discrete time: innovation ≈ (grad - self.m[i])
            self.innovation[i] = grad - self.m[i]

            # Update first moment (filtered gradient) with dissipation
            # This is the "quantum filtering" aspect
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update second moment (uncertainty measure)
            # Incorporates measurement backaction
            self.v[i] = (self.beta2 * self.v[i] +
                        (1 - self.beta2) * (grad ** 2) +
                        self.gamma * self.innovation[i] ** 2)

            # Bias correction (from Adam)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Belavkin-style update combining:
            # 1. Hamiltonian flow: -∇L term
            # 2. Lindblad dissipation: uncertainty-weighted
            # 3. Measurement backaction: innovation-driven adaptation

            # Adaptive learning rate based on uncertainty
            adaptive_lr = self.lr * self.alpha / (torch.sqrt(v_hat) + 1e-8)

            # Parameter update with innovation term
            param.data -= (adaptive_lr * m_hat +
                          self.gamma * self.innovation[i] / (torch.sqrt(v_hat) + 1e-8))

class BelOptimSecondOrder:
    def __init__(self, params, lr=0.01, gamma=0.1, fisher_alpha=0.01, beta1=0.9, beta2=0.999):
        self.params = list(params)
        self.lr = lr
        self.gamma = gamma
        self.fisher_alpha = fisher_alpha
        self.beta1 = beta1
        self.beta2 = beta2

        # Extended state
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.fisher_diag = [torch.ones_like(p) for p in self.params]  # Diagonal Fisher approximation
        self.innovation_history = [deque(maxlen=10) for _ in self.params]

    def compute_fisher_diag(self, gradients):
        """Approximate diagonal Fisher information from gradient history"""
        for i, grad in enumerate(gradients):
            self.fisher_diag[i] = (self.fisher_alpha * grad ** 2 +
                                   (1 - self.fisher_alpha) * self.fisher_diag[i])

    def step(self, gradients, hessian_vector_product=None):
        self.compute_fisher_diag(gradients)

        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            # Innovation with Fisher metric
            innovation = grad - self.m[i]
            self.innovation_history[i].append(innovation)

            # Filtered gradient with Fisher preconditioning
            self.m[i] = self.m[i] + (1 - self.beta1) * innovation / (self.fisher_diag[i] + 1e-8)

            # Uncertainty evolution with measurement backaction
            innovation_var = torch.var(torch.stack(list(self.innovation_history[i]))) if len(self.innovation_history[i]) > 1 else torch.zeros_like(param)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2 + self.gamma * innovation_var)

            # Natural gradient direction (Fisher-weighted)
            natural_grad = self.m[i] / (self.fisher_diag[i] + 1e-8)

            # Belavkin update with curvature
            param.data -= self.lr * natural_grad / (torch.sqrt(self.v[i]) + 1e-8)
