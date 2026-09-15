import numpy as np
import torch

from neuralnetwork import NeuralNetwork


def max_relative_error(a, b):
    denominator = np.maximum(1e-12, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denominator))


def relative_error(a, b):
    return float(abs(a - b) / max(1e-12, abs(a) + abs(b)))


def check_autograd():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(8, 64)).astype(np.float64)
    y = rng.integers(0, 10, size=8)

    model = NeuralNetwork(64, 32, 10, seed=999)
    manual_loss, _, manual_grads = model.loss_and_backward(X, y)

    W1 = torch.tensor(model.W1, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(model.b1, dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(model.W2, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(model.b2, dtype=torch.float64, requires_grad=True)
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.long)

    Z1 = X_t @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2
    loss = torch.nn.functional.cross_entropy(Z2, y_t)
    loss.backward()

    reference = {
        "W1": W1.grad.detach().numpy(),
        "b1": b1.grad.detach().numpy(),
        "W2": W2.grad.detach().numpy(),
        "b2": b2.grad.detach().numpy(),
    }

    print("Autograd check")
    print(f"  manual loss  = {manual_loss:.12f}")
    print(f"  torch loss   = {loss.item():.12f}")

    passed = True
    for name in ("W1", "b1", "W2", "b2"):
        error = max_relative_error(manual_grads[name], reference[name])
        print(f"  {name}: relative error = {error:.3e}")
        passed = passed and error < 1e-6
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_numerical():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(5, 64)).astype(np.float64)
    y = rng.integers(0, 10, size=5)

    model = NeuralNetwork(64, 32, 10, seed=555)
    _, _, analytical = model.loss_and_backward(X, y)
    h = 1e-5
    checks = [("W1", (0, 0)), ("W1", (10, 3)), ("b1", (2,)), ("W2", (3, 7)), ("b2", (4,))]

    print("\nNumerical gradient check")
    passed = True

    for name, index in checks:
        parameter = getattr(model, name)
        original = parameter[index]

        parameter[index] = original + h
        loss_plus, _, _ = model.loss_and_backward(X, y)

        parameter[index] = original - h
        loss_minus, _, _ = model.loss_and_backward(X, y)
        parameter[index] = original

        numerical = (loss_plus - loss_minus) / (2 * h)
        analytic = analytical[name][index]
        error = relative_error(analytic, numerical)

        print(
            f"  {name}{index}: analytical={analytic:.8e}, "
            f"numerical={numerical:.8e}, error={error:.3e}"
        )
        passed = passed and error < 1e-5

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("Running gradient correctness harness")
    print("=" * 42)

    autograd_passed = check_autograd()
    numerical_passed = check_numerical()

    overall = autograd_passed and numerical_passed
    print("\nOverall result:", "PASS" if overall else "FAIL")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
