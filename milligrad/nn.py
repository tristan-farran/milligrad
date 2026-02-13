import numpy as np
from .milligrad import Tensor


class Linear:

    def __init__(self, nin, nout, nonlin=True):
        self.w = Tensor(np.random.randn(nin, nout) * (2 / nin) ** 0.5)
        self.b = Tensor(np.zeros(nout))
        self.nonlin = nonlin

    def __call__(self, x):
        act = x @ self.w + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        nin, nout = self.w.data.shape
        return f"{'ReLU' if self.nonlin else 'Linear'}Linear({nin}, {nout})"


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Linear(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0)

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
