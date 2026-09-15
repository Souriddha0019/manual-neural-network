# Manual Two-Layer Neural Network — Write-up

## 1. Objective and approach

The goal of this project was to implement a small feedforward neural network using basic matrix operations instead of relying on a high-level neural-network framework for the model itself. I used NumPy for the complete forward pass, loss calculation, backward pass, and parameter updates. PyTorch was used separately as a trusted reference for gradient checking.

The network has an input layer of 64 features, one hidden layer with 32 neurons, and an output layer with 10 neurons. The 64 inputs come from the `8 x 8` images in the scikit-learn Digits dataset. A ReLU activation is used after the first linear layer, followed by a softmax output so the network can predict one of the ten digits.

The forward pass is:

\[
Z_1 = XW_1 + b_1
\]

\[
A_1 = \operatorname{ReLU}(Z_1)
\]

\[
Z_2 = A_1W_2 + b_2
\]

\[
P = \operatorname{softmax}(Z_2)
\]

The loss is the mean cross-entropy loss:

\[
L = -\frac{1}{N}\sum_{i=1}^{N}\log P_{i,y_i}
\]

He initialization was used for the weight matrices because the hidden layer uses ReLU.

## 2. Manual backward pass

The main purpose of the implementation was to derive the gradients using the chain rule and then translate those equations directly into matrix operations.

For the combined softmax and cross-entropy output, the derivative is particularly simple:

\[
\frac{\partial L}{\partial Z_2} = \frac{P-Y}{N}
\]

where `Y` is the one-hot encoded target matrix and `N` is the mini-batch size.

For the second linear layer,

\[
Z_2 = A_1W_2+b_2,
\]

so:

\[
dW_2 = A_1^T dZ_2
\]

\[
db_2 = \sum dZ_2
\]

\[
dA_1 = dZ_2W_2^T
\]

The ReLU derivative is an element-wise mask. It is 1 when the corresponding value of `Z1` is positive and 0 otherwise:

\[
dZ_1 = dA_1 \odot \mathbf{1}[Z_1>0]
\]

Finally, for the first linear layer,

\[
Z_1=XW_1+b_1,
\]

which gives:

\[
dW_1=X^T dZ_1
\]

\[
db_1=\sum dZ_1
\]

The implementation also computes `dX = dZ1 W1^T`, although only parameter gradients are needed for the training update.

## 3. Correctness testing

A separate file, `test_gradients.py`, acts as the correctness harness. This was kept outside the main implementation so that the reference calculation is clearly independent from the code being tested.

The first test recreates the same network in PyTorch and calls `loss.backward()`. The manually calculated NumPy gradients are then compared with PyTorch's gradients using the maximum relative error. The harness checks `W1`, `b1`, `W2`, and `b2` separately.

The second test uses numerical differentiation. For selected parameter values, the code evaluates the loss at `theta + h` and `theta - h` and estimates the derivative with:

\[
\frac{\partial L}{\partial \theta}\approx
\frac{L(\theta+h)-L(\theta-h)}{2h}
\]

The manual and numerical values are then compared with a relative-error threshold.

Using both methods is useful because the tests fail for different reasons. Autograd checks the complete symbolic implementation against another automatic differentiation system, while finite differences provide a direct local approximation of the derivative.

The harness prints `PASS` or `FAIL` for each check and exits with status code 0 only when every check passes.

## 4. Training experiment

The real task uses the scikit-learn Digits dataset. Each sample is an 8-by-8 grayscale image, flattened to 64 features and scaled from approximately 0–16 to 0–1. The data is split into training and test sets using an 80/20 stratified split.

Training uses mini-batch gradient descent with a batch size of 64, a learning rate of 0.1, and 100 epochs. The parameter update is simply:

\[
\theta \leftarrow \theta-\eta\nabla_\theta L
\]

where `eta` is the learning rate.

In a representative run, the loss decreased from roughly 2.26 at the beginning of training to about 0.06 after 100 epochs. Test accuracy reached roughly 95–96%. The decrease in loss and the improvement in accuracy show that the manual forward and backward passes are sufficient to train the network on a real classification problem.

## 5. Gradient mistakes and how they were found

The most important mistake to watch for is forgetting the `1/N` factor in the softmax-cross-entropy gradient. The correct expression is `(P-Y)/N`. Leaving out the division causes all gradients to be scaled by the batch size. This can still produce apparent learning with a different learning rate, so the error is easier to detect with an independent gradient check than from training behavior alone.

A second common mistake is using the ReLU derivative incorrectly. The backward pass must block gradients where `Z1` is non-positive. Using an incorrect mask changes which hidden units receive gradients.

Matrix orientation is another likely source of errors. For example, the second-layer weight gradient must be `A1.T @ dZ2`, while the first-layer weight gradient must be `X.T @ dZ1`. Checking the expected shapes is a simple way to catch these errors.

The debugging process was to verify the forward-pass shapes first, then compare every parameter gradient against PyTorch autograd, and finally check a few individual entries with finite differences. Training was only considered meaningful after both correctness checks passed.

## 6. Conclusion

The final project separates the model implementation from the correctness harness and documentation. The neural network performs all of its learning calculations with explicit NumPy matrix operations, while the independent tests provide evidence that the chain-rule derivatives are correct. The real-data experiment then shows that the implementation is not only mathematically consistent but also able to learn a practical classification task.
