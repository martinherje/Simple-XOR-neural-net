import numpy as np

# ------- data (xor) ------
X = np.array([
    [0.0, 0.0],
    [0.0, 0.1],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=np.float32)

y = np.array([[0.0], [1.0], [1.0], [0.0],], dtype=np.float32)

# ------ helpers ------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def bce_loss(y_pred, y_true, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1- eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


# --------- model: 2 - hidden - 1 -----------
rng = np.random.default_rng(0)
hidden = 8

W1 = rng.normal(0, 1, size=(2, hidden)).astype(np.float32) * 0.5
b1 = np.zeros((1, hidden), dtype=np.float32)

W2 = rng.normal(0, 1, size=(hidden, 1)).astype(np.float32) * 0.5
b2 = np.zeros((1, 1), dtype=np.float32)

lr = 0.3
steps = 20000

for step in range(steps):
    # ------ forward pass ----- 
    z1 = X @ W1 + b1            # (N, hidden)
    a1 = np.tanh(z1)            # nonlinearity
    z2 = a1 @ W2 + b2           # (N, 1)
    yhat = sigmoid(z2)          # probabilities

    loss = bce_loss(yhat, y)


    # ----- backward pass (backpropagation) -----
    # BCE with sigmoid: dL(dz2 = yhat -y) / N
    N = X.shape[0]
    dz2 = (yhat - y) / N        # (N, 1)

    dW2 = a1.T @ dz2            # (hidden, 1)
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ W2.T            # (N, hidden)
    dz1 = da1 * (1 - np.tanh(z1) ** 2) # tanh' = 1 - tanh^2

    dW1 = X.T @ dz1             # (2, hidden)
    db1 = dz1.sum(axis=0, keepdims=True)


    #------ SGD update ------

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if step % 500 == 0:
        print(f"step {step:4d} loss {loss:.4f}")


# ------ results ------
print("\nPredictions:")
print(np.hstack([X, yhat]))
print("\nRounded:", (yhat > 0.5).astype(int).ravel())




