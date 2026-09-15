import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.default_rng(seed)

        # He initialization works well with ReLU
        self.W1 = (
            rng.standard_normal((input_size, hidden_size))
            * np.sqrt(2 / input_size)
        )
        self.b1 = np.zeros(hidden_size)

        self.W2 = (
            rng.standard_normal((hidden_size, output_size))
            * np.sqrt(2 / hidden_size)
        )
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        # Subtracting the maximum keeps the exponentials stable
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # First linear layer
        Z1 = X @ self.W1 + self.b1

        # ReLU activation
        A1 = self.relu(Z1)

        # Second linear layer
        Z2 = A1 @ self.W2 + self.b2

        # Convert scores into probabilities
        P = self.softmax(Z2)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "P": P
        }

        return P, cache

    def loss_and_backward(self, X, y):
        P, cache = self.forward(X)
        n = X.shape[0]

        # Cross-entropy loss
        correct_probs = P[np.arange(n), y]
        loss = -np.mean(
            np.log(np.clip(correct_probs, 1e-12, 1.0))
        )

        # One-hot encoded labels
        Y = np.zeros_like(P)
        Y[np.arange(n), y] = 1

        # For softmax + cross entropy:
        # dZ2 = (P - Y) / n
        dZ2 = (P - Y) / n

        # Z2 = A1 @ W2 + b2
        dW2 = cache["A1"].T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.W2.T

        # ReLU derivative
        dZ1 = dA1 * (cache["Z1"] > 0)

        # Z1 = X @ W1 + b1
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
        dX = dZ1 @ self.W1.T

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "X": dX
        }

        return loss, P, grads

    def predict(self, X):
        P, _ = self.forward(X)
        return np.argmax(P, axis=1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


def relative_error(a, b):
    return abs(a - b) / max(1e-12, abs(a) + abs(b))


def max_relative_error(a, b):
    denominator = np.maximum(
        1e-12,
        np.abs(a) + np.abs(b)
    )
    return np.max(
        np.abs(a - b) / denominator
    )


def check_autograd():
    print("\nChecking gradients with PyTorch...")

    rng = np.random.default_rng(123)

    X = rng.normal(size=(8, 64))
    y = rng.integers(0, 10, size=8)

    model = NeuralNetwork(64, 32, 10, seed=999)

    manual_loss, _, manual_grads = model.loss_and_backward(X, y)

    W1 = torch.tensor(
        model.W1,
        dtype=torch.float64,
        requires_grad=True
    )

    b1 = torch.tensor(
        model.b1,
        dtype=torch.float64,
        requires_grad=True
    )

    W2 = torch.tensor(
        model.W2,
        dtype=torch.float64,
        requires_grad=True
    )

    b2 = torch.tensor(
        model.b2,
        dtype=torch.float64,
        requires_grad=True
    )

    X_torch = torch.tensor(
        X,
        dtype=torch.float64
    )

    y_torch = torch.tensor(
        y,
        dtype=torch.long
    )

    # Same network written using PyTorch
    Z1 = X_torch @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2

    loss = torch.nn.functional.cross_entropy(
        Z2,
        y_torch
    )

    loss.backward()

    torch_grads = {
        "W1": W1.grad.detach().numpy(),
        "b1": b1.grad.detach().numpy(),
        "W2": W2.grad.detach().numpy(),
        "b2": b2.grad.detach().numpy()
    }

    print(f"Manual loss : {manual_loss:.10f}")
    print(f"PyTorch loss: {loss.item():.10f}\n")

    passed = True

    for name in ["W1", "b1", "W2", "b2"]:
        error = max_relative_error(
            manual_grads[name],
            torch_grads[name]
        )

        print(
            f"{name} relative error: {error:.3e}"
        )

        if error > 1e-6:
            passed = False

    if passed:
        print("Autograd gradient check passed.")
    else:
        print("Autograd gradient check failed.")

    return passed


def numerical_gradient_check():
    print("\nChecking a few gradients numerically...")

    rng = np.random.default_rng(456)

    X = rng.normal(size=(5, 64))
    y = rng.integers(0, 10, size=5)

    model = NeuralNetwork(64, 32, 10, seed=555)

    _, _, analytical_grads = model.loss_and_backward(
        X,
        y
    )

    h = 1e-5

    tests = [
        ("W1", (0, 0)),
        ("W1", (10, 3)),
        ("b1", (2,)),
        ("W2", (3, 7)),
        ("b2", (4,))
    ]

    passed = True

    for name, index in tests:
        parameter = getattr(model, name)
        original = parameter[index]

        # f(x + h)
        parameter[index] = original + h
        loss_plus, _, _ = model.loss_and_backward(X, y)

        # f(x - h)
        parameter[index] = original - h
        loss_minus, _, _ = model.loss_and_backward(X, y)

        # Restore original value
        parameter[index] = original

        # Central difference:
        # f'(x) ≈ [f(x+h) - f(x-h)] / 2h
        numerical = (
            loss_plus - loss_minus
        ) / (2 * h)

        analytical = analytical_grads[name][index]

        error = relative_error(
            analytical,
            numerical
        )

        print(
            f"{name}{index}: "
            f"analytical={analytical:.8e}, "
            f"numerical={numerical:.8e}, "
            f"error={error:.3e}"
        )

        if error > 1e-5:
            passed = False

    if passed:
        print("Numerical gradient check passed.")
    else:
        print("Numerical gradient check failed.")

    return passed


def load_data():
    digits = load_digits()

    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)

    # Pixel values are between 0 and 16
    X /= 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train():
    X_train, X_test, y_train, y_test = load_data()

    model = NeuralNetwork(
        input_size=64,
        hidden_size=32,
        output_size=10,
        seed=42
    )

    epochs = 100
    batch_size = 64
    learning_rate = 0.1

    rng = np.random.default_rng(42)

    initial_loss = None

    for epoch in range(1, epochs + 1):

        # Shuffle the training data
        order = rng.permutation(len(X_train))

        X_train_shuffled = X_train[order]
        y_train_shuffled = y_train[order]

        losses = []

        for start in range(
            0,
            len(X_train),
            batch_size
        ):
            end = start + batch_size

            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward pass + manual backward pass
            loss, _, grads = model.loss_and_backward(
                X_batch,
                y_batch
            )

            losses.append(loss)

            # Gradient descent
            model.W1 -= learning_rate * grads["W1"]
            model.b1 -= learning_rate * grads["b1"]
            model.W2 -= learning_rate * grads["W2"]
            model.b2 -= learning_rate * grads["b2"]

        avg_loss = np.mean(losses)

        if initial_loss is None:
            initial_loss = avg_loss

        if epoch == 1 or epoch % 10 == 0:
            train_accuracy = model.accuracy(
                X_train,
                y_train
            )

            test_accuracy = model.accuracy(
                X_test,
                y_test
            )

            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train accuracy: "
                f"{train_accuracy * 100:.2f}% | "
                f"Test accuracy: "
                f"{test_accuracy * 100:.2f}%"
            )

    final_loss = avg_loss

    final_train_accuracy = model.accuracy(
        X_train,
        y_train
    )

    final_test_accuracy = model.accuracy(
        X_test,
        y_test
    )

    print("\nTraining finished.")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(
        f"Final train accuracy: "
        f"{final_train_accuracy * 100:.2f}%"
    )
    print(
        f"Final test accuracy: "
        f"{final_test_accuracy * 100:.2f}%"
    )

    return model


def main():
    print("Two-layer neural network")

    # Check the backward pass before training
    if not check_autograd():
        print("\nStopping because the gradient check failed.")
        return

    if not numerical_gradient_check():
        print("\nStopping because the numerical check failed.")
        return

    print("\nTraining...\n")
    train()

    print("\nGradient mistakes to watch for:")
    print("- Forgetting the 1/n factor in dZ2.")
    print("- Using the wrong ReLU derivative.")
    print("- Mixing up matrix dimensions.")
    print("- Updating a parameter with the wrong gradient.")


if __name__ == "__main__":
    main()