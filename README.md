A small feedforward neural network built with basic NumPy matrix operations. The network has two linear layers with a ReLU activation between them and is trained on the scikit-learn Digits dataset.

## Project structure

```text
manual-neural-network/
├── neuralnetwork.py
├── test_gradients.py
├── README.md
├── WRITEUP.md
├── requirements.txt
└── .gitignore
```

## Requirements

Python 3.10+ is recommended. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run the correctness harness

The harness compares the manually derived gradients against PyTorch autograd and a finite-difference numerical gradient.

```bash
python test_gradients.py
```

A successful run ends with:

```text
Overall result: PASS
```

The script returns exit code `0` on success and `1` on failure, so it can also be used in CI.

## Train the network

```bash
python neuralnetwork.py
```

The program trains a `64 -> 32 -> 10` network on the Digits dataset and prints the loss and train/test accuracy during training.

## How the network works

For an input batch `X`:

```text
Z1 = XW1 + b1
A1 = ReLU(Z1)
Z2 = A1W2 + b2
P  = softmax(Z2)
```

The loss is mean cross-entropy. The main backward-pass equations are:

```text
dZ2 = (P - Y) / N

dW2 = A1.T @ dZ2
db2 = sum(dZ2)
dA1 = dZ2 @ W2.T

dZ1 = dA1 * (Z1 > 0)

dW1 = X.T @ dZ1
db1 = sum(dZ1)
```

The implementation performs these calculations directly with NumPy rather than using an automatic differentiation library.

## Reproducibility

The code uses fixed random seeds for initialization, shuffling, and the gradient-check inputs, so the main results are reproducible on the same software environment.

## Git workflow

After cloning the repository, create a virtual environment if desired, install the requirements, run the correctness harness, and then train the network. Changes should be committed in small, meaningful steps rather than as one large commit.
