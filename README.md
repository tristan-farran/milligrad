# Milligrad

A minimal autograd engine with n-dimensional tensor support. Milligrad is inspired by [micrograd](https://github.com/karpathy/micrograd) but extends it to handle multidimensional arrays and matrix operations, making it suitable for implementing neural networks.

Installation
------------
```bash
pip install -e .
```

Quick start
-----------
Train a simple neural network:

```python
from milligrad import Tensor
from milligrad.nn import MLP

# Create a simple MLP
model = MLP(2, [16, 16, 1])

# Training data
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = Tensor([[0], [1], [1], [0]])

# Training loop
for step in range(100):
    # Forward pass
    pred = model(X)
    loss = ((pred - y) ** 2).sum()
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update weights
    for p in model.parameters():
        p.data -= 0.1 * p.grad
```

How it works
------------
Milligrad implements reverse-mode automatic differentiation (backpropagation) by building a computational graph dynamically:

1. **Forward pass**: Operations on `Tensor` objects store their inputs and operation type
2. **Backward pass**: Call `.backward()` on the output to recursively compute gradients
3. **Gradient accumulation**: Gradients flow backward through the graph using the chain rule

Each tensor stores:
- `data`: The n-dimensional array value
- `grad`: The accumulated gradient
- `_prev`: Set of parent tensors in the computation graph
- `_backward`: Function to propagate gradients to parents

Supported operations include addition, multiplication, matrix multiplication, power, ReLU, sum, and more.