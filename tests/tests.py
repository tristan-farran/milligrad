"""Tests for Milligrad autograd engine."""

import numpy as np
import pytest
from milligrad import Tensor
from milligrad.nn import Linear, MLP


class TestTensorOperations:
    """Test basic tensor operations."""

    def test_addition(self):
        """Test addition operation and gradient."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a + b
        c.backward()

        np.testing.assert_array_equal(c.data, [5.0, 7.0, 9.0])
        np.testing.assert_array_equal(a.grad, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(b.grad, [1.0, 1.0, 1.0])

    def test_addition_with_scalar(self):
        """Test addition with scalar."""
        a = Tensor([1.0, 2.0, 3.0])
        b = a + 5.0
        b.backward()

        np.testing.assert_array_equal(b.data, [6.0, 7.0, 8.0])
        np.testing.assert_array_equal(a.grad, [1.0, 1.0, 1.0])

    def test_subtraction(self):
        """Test subtraction operation and gradient."""
        a = Tensor([5.0, 7.0, 9.0])
        b = Tensor([1.0, 2.0, 3.0])
        c = a - b
        c.backward()

        np.testing.assert_array_equal(c.data, [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(a.grad, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(b.grad, [-1.0, -1.0, -1.0])

    def test_multiplication(self):
        """Test multiplication operation and gradient."""
        a = Tensor([2.0, 3.0, 4.0])
        b = Tensor([5.0, 6.0, 7.0])
        c = a * b
        c.backward()

        np.testing.assert_array_equal(c.data, [10.0, 18.0, 28.0])
        np.testing.assert_array_equal(a.grad, [5.0, 6.0, 7.0])
        np.testing.assert_array_equal(b.grad, [2.0, 3.0, 4.0])

    def test_division(self):
        """Test division operation and gradient."""
        a = Tensor([12.0, 18.0, 24.0])
        b = Tensor([3.0, 2.0, 4.0])
        c = a / b
        c.backward()

        np.testing.assert_array_equal(c.data, [4.0, 9.0, 6.0])
        np.testing.assert_allclose(a.grad, [1 / 3, 1 / 2, 1 / 4])
        # dc/db = -a/b^2
        np.testing.assert_allclose(b.grad, [-12.0 / 9, -18.0 / 4, -24.0 / 16])

    def test_power(self):
        """Test power operation and gradient."""
        a = Tensor([2.0, 3.0, 4.0])
        b = a**2
        b.backward()

        np.testing.assert_array_equal(b.data, [4.0, 9.0, 16.0])
        np.testing.assert_array_equal(a.grad, [4.0, 6.0, 8.0])

    def test_matmul(self):
        """Test matrix multiplication and gradient."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        c.backward()

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_equal(c.data, expected)
        np.testing.assert_array_equal(a.grad, [[11.0, 15.0], [11.0, 15.0]])
        np.testing.assert_array_equal(b.grad, [[4.0, 4.0], [6.0, 6.0]])

    def test_negation(self):
        """Test negation operation and gradient."""
        a = Tensor([1.0, -2.0, 3.0])
        b = -a
        b.backward()

        np.testing.assert_array_equal(b.data, [-1.0, 2.0, -3.0])
        np.testing.assert_array_equal(a.grad, [-1.0, -1.0, -1.0])

    def test_reverse_operations(self):
        """Test reverse operations like radd, rsub, rmul."""
        a = Tensor([1.0, 2.0, 3.0])

        # radd
        b = 5.0 + a
        np.testing.assert_array_equal(b.data, [6.0, 7.0, 8.0])

        # rsub
        c = 10.0 - a
        np.testing.assert_array_equal(c.data, [9.0, 8.0, 7.0])

        # rmul
        d = 2.0 * a
        np.testing.assert_array_equal(d.data, [2.0, 4.0, 6.0])

        # rtruediv
        e = 12.0 / Tensor([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(e.data, [6.0, 4.0, 3.0])


class TestBroadcasting:
    """Test broadcasting support."""

    def test_broadcast_addition(self):
        """Test addition with broadcasting."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = Tensor([10.0, 20.0, 30.0])
        c = a + b
        c.backward()

        expected = np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
        np.testing.assert_array_equal(c.data, expected)
        np.testing.assert_array_equal(a.grad, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(b.grad, [2.0, 2.0, 2.0])

    def test_broadcast_multiplication(self):
        """Test multiplication with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = Tensor([[10.0, 20.0]])
        c = a * b
        c.backward()

        expected = np.array([[10.0, 40.0], [30.0, 80.0], [50.0, 120.0]])
        np.testing.assert_array_equal(c.data, expected)
        # Gradient sums over broadcast axis: 1+3+5=9, 2+4+6=12
        np.testing.assert_array_equal(b.grad, [[9.0, 12.0]])

    def test_broadcast_with_ones(self):
        """Test broadcasting with singleton dimensions."""
        a = Tensor([[[1.0, 2.0]], [[3.0, 4.0]]])  # shape (2, 1, 2)
        b = Tensor([[5.0], [6.0]])  # shape (2, 1)
        c = a * b
        c.backward()

        # Shape should be (2, 2, 2)
        assert c.data.shape == (2, 2, 2)
        # Check gradient shapes match original
        assert a.grad.shape == a.data.shape
        assert b.grad.shape == b.data.shape


class TestActivations:
    """Test activation functions."""

    def test_relu_forward(self):
        """Test ReLU forward pass."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        b = a.relu()

        np.testing.assert_array_equal(b.data, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_relu_backward(self):
        """Test ReLU backward pass."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        b = a.relu()
        b.backward()

        np.testing.assert_array_equal(a.grad, [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_log_softmax_forward(self):
        """Test log_softmax forward pass."""
        a = Tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0]])
        b = a.log_softmax(axis=1)

        # Check output is valid log probabilities
        assert np.all(b.data <= 0)
        # Check each row sums to 1 in probability space
        probs = np.exp(b.data)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])

    def test_log_softmax_backward(self):
        """Test log_softmax backward pass with loss."""
        # Simulate classification scenario
        logits = Tensor([[2.0, 1.0, 0.1]])
        target = np.array([[1.0, 0.0, 0.0]])  # One-hot encoded

        log_probs = logits.log_softmax(axis=1)
        loss = -(Tensor(target) * log_probs).sum()
        loss.backward()

        # Gradient should exist and have correct shape
        assert logits.grad.shape == logits.data.shape
        # Gradient should not be all zeros
        assert np.any(logits.grad != 0)


class TestSum:
    """Test sum operation."""

    def test_sum_all(self):
        """Test summing all elements."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a.sum()
        b.backward()

        assert b.data == 21.0
        np.testing.assert_array_equal(a.grad, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_sum_axis(self):
        """Test summing along an axis."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a.sum(axis=1)
        b.backward()

        np.testing.assert_array_equal(b.data, [6.0, 15.0])
        np.testing.assert_array_equal(a.grad, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_sum_keepdims(self):
        """Test sum with keepdims."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a.sum(axis=1, keepdims=True)
        b.backward()

        assert b.data.shape == (2, 1)
        np.testing.assert_array_equal(b.data, [[6.0], [15.0]])


class TestBackpropagation:
    """Test complex backpropagation scenarios."""

    def test_chain_rule(self):
        """Test chain rule with multiple operations."""
        a = Tensor([2.0])
        b = Tensor([3.0])
        c = a * b  # 6.0
        d = c + a  # 8.0
        e = d * d  # 64.0
        e.backward()

        # de/da = de/dd * dd/da = 2*d * (dc/da + 1) = 2*8 * (3 + 1) = 64
        np.testing.assert_allclose(a.grad, [64.0])
        # de/db = de/dd * dd/dc * dc/db = 2*d * 1 * a = 2*8*2 = 32
        np.testing.assert_allclose(b.grad, [32.0])

    def test_multiple_paths(self):
        """Test gradient accumulation through multiple paths."""
        a = Tensor([2.0])
        b = a + a
        c = a * a
        d = b + c  # (a + a) + (a * a) = 2a + a^2
        d.backward()

        # dd/da = 2 + 2a = 2 + 4 = 6
        np.testing.assert_allclose(a.grad, [6.0])

    def test_complex_computation_graph(self):
        """Test complex computation graph."""
        x = Tensor([3.0])
        y = Tensor([4.0])

        # f(x, y) = (x + y) * (x - y)
        z1 = x + y  # 7
        z2 = x - y  # -1
        out = z1 * z2  # -7
        out.backward()

        # df/dx = (x-y) + (x+y) = 2x = 6
        # df/dy = (x-y)*1 + (x+y)*(-1) = -2y = -8
        np.testing.assert_allclose(x.grad, [6.0])
        np.testing.assert_allclose(y.grad, [-8.0])


class TestLinearLayer:
    """Test Linear layer."""

    def test_linear_creation(self):
        """Test Linear layer initialization."""
        layer = Linear(10, 5)

        assert layer.w.data.shape == (10, 5)
        assert layer.b.data.shape == (5,)
        assert layer.nonlin is True

    def test_linear_forward(self):
        """Test Linear layer forward pass."""
        layer = Linear(3, 2, nonlin=False)
        layer.w.data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        layer.b.data = np.array([0.5, 1.0])

        x = Tensor([[1.0, 2.0, 3.0]])
        out = layer(x)

        # [1, 2, 3] @ [[1, 2], [3, 4], [5, 6]] + [0.5, 1.0]
        # = [1+6+15, 2+8+18] + [0.5, 1.0] = [22.5, 29.0]
        np.testing.assert_allclose(out.data, [[22.5, 29.0]])

    def test_linear_with_relu(self):
        """Test Linear layer with ReLU activation."""
        layer = Linear(2, 2, nonlin=True)
        layer.w.data = np.array([[1.0, -1.0], [-1.0, 1.0]])
        layer.b.data = np.array([0.0, 0.0])

        x = Tensor([[2.0, 3.0]])
        out = layer(x)

        # [2, 3] @ [[1, -1], [-1, 1]] = [-1, 1]
        # ReLU: [0, 1]
        np.testing.assert_allclose(out.data, [[0.0, 1.0]])

    def test_linear_parameters(self):
        """Test Linear layer parameters method."""
        layer = Linear(3, 2)
        params = layer.parameters()

        assert len(params) == 2
        assert params[0] is layer.w
        assert params[1] is layer.b

    def test_linear_backward(self):
        """Test Linear layer backward pass."""
        layer = Linear(2, 1, nonlin=False)
        layer.w.data = np.array([[2.0], [3.0]])
        layer.b.data = np.array([1.0])

        x = Tensor([[1.0, 2.0]])
        out = layer(x)
        out.backward()

        # Check gradients exist
        assert np.any(x.grad != 0)
        assert np.any(layer.w.grad != 0)
        assert np.any(layer.b.grad != 0)


class TestMLP:
    """Test MLP network."""

    def test_mlp_creation(self):
        """Test MLP initialization."""
        model = MLP(10, [5, 3, 2])

        assert len(model.layers) == 3
        assert model.layers[0].w.data.shape == (10, 5)
        assert model.layers[1].w.data.shape == (5, 3)
        assert model.layers[2].w.data.shape == (3, 2)
        # Last layer should not have nonlinearity
        assert model.layers[2].nonlin is False

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP(2, [3, 1])
        x = Tensor([[1.0, 2.0]])
        out = model(x)

        # Should produce output of shape (1, 1)
        assert out.data.shape == (1, 1)

    def test_mlp_parameters(self):
        """Test MLP parameters method."""
        model = MLP(3, [5, 2])
        params = model.parameters()

        # Should have 4 parameters (2 layers * 2 params per layer)
        assert len(params) == 4

    def test_mlp_zero_grad(self):
        """Test MLP zero_grad method."""
        model = MLP(2, [3, 1])
        x = Tensor([[1.0, 2.0]])
        out = model(x)
        # Use sum() to ensure gradients flow to all parameters
        loss = out.sum()
        loss.backward()

        # At least some gradients should be non-zero
        has_nonzero = any(np.any(p.grad != 0) for p in model.parameters())
        assert has_nonzero, "Expected at least some gradients to be non-zero"

        # Zero gradients
        model.zero_grad()

        # All gradients should be zero
        for p in model.parameters():
            np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))

    def test_mlp_training_step(self):
        """Test a simple MLP training step."""
        model = MLP(2, [4, 1])

        # Simple dataset: XOR problem
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[0.0], [1.0], [1.0], [0.0]])

        # One training step
        model.zero_grad()
        out = model(Tensor(X))
        loss = ((out - Tensor(y)) ** 2).sum()
        loss.backward()

        # Check that gradients were computed
        for p in model.parameters():
            assert np.any(p.grad != 0)

        # Store old parameters
        old_params = [p.data.copy() for p in model.parameters()]

        # Update parameters
        lr = 0.01
        for p in model.parameters():
            p.data -= lr * p.grad

        # Parameters should have changed
        for old, new in zip(old_params, model.parameters()):
            assert not np.allclose(old, new.data)


class TestNumericalGradients:
    """Test gradients against numerical approximation."""

    def numerical_gradient(self, f, x, eps=1e-5):
        """Compute numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_flat = x.flatten()
            old_val = x_flat[i]

            x_flat[i] = old_val + eps
            f_plus = f(x_flat.reshape(x.shape))

            x_flat[i] = old_val - eps
            f_minus = f(x_flat.reshape(x.shape))

            x_flat[i] = old_val
            grad.flat[i] = (f_plus - f_minus) / (2 * eps)

        return grad

    def test_numerical_gradient_add(self):
        """Test addition gradient against numerical gradient."""

        def f(x):
            a = Tensor(x)
            b = Tensor([2.0, 3.0])
            c = a + b
            return c.sum().data

        x = np.array([1.0, 2.0])
        a = Tensor(x)
        b = Tensor([2.0, 3.0])
        c = a + b
        c.sum().backward()

        num_grad = self.numerical_gradient(f, x)
        np.testing.assert_allclose(a.grad, num_grad, rtol=1e-4)

    def test_numerical_gradient_mul(self):
        """Test multiplication gradient against numerical gradient."""

        def f(x):
            a = Tensor(x)
            b = Tensor([2.0, 3.0])
            c = a * b
            return c.sum().data

        x = np.array([1.0, 2.0])
        a = Tensor(x)
        b = Tensor([2.0, 3.0])
        c = a * b
        c.sum().backward()

        num_grad = self.numerical_gradient(f, x)
        np.testing.assert_allclose(a.grad, num_grad, rtol=1e-4)

    def test_numerical_gradient_matmul(self):
        """Test matmul gradient against numerical gradient."""

        def f(x):
            a = Tensor(x.reshape(2, 2))
            b = Tensor([[1.0, 2.0], [3.0, 4.0]])
            c = a @ b
            return c.sum().data

        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = Tensor(x)
        b = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = a @ b
        c.sum().backward()

        num_grad = self.numerical_gradient(f, x)
        np.testing.assert_allclose(a.grad, num_grad, rtol=1e-4)

    def test_numerical_gradient_relu(self):
        """Test ReLU gradient against numerical gradient."""

        def f(x):
            a = Tensor(x)
            b = a.relu()
            return b.sum().data

        # Avoid x=0 where ReLU is not differentiable
        x = np.array([-2.0, -1.0, 1.0, 2.0])
        a = Tensor(x)
        b = a.relu()
        b.sum().backward()

        num_grad = self.numerical_gradient(f, x)
        np.testing.assert_allclose(a.grad, num_grad, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
