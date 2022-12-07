"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            grad = param.grad + self.weight_decay * param.data
            u_next = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * grad
            self.u[param] = u_next.detach()
            param.data = param - self.lr * u_next

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(
            np.array(
                [
                    np.linalg.norm(p.grad.detach().numpy()).reshape((1,))
                    for p in self.params
                ]
            )
        )
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min(clip_coef, 1.0)
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            grad = param.grad + self.weight_decay * param.data
            m_next = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            v_next = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * grad**2
            self.m[param] = m_next.detach()
            self.v[param] = v_next.detach()
            m_next /= 1 - self.beta1**self.t
            v_next /= 1 - self.beta2**self.t
            param.data = param - self.lr * m_next / (v_next**0.5 + self.eps)
