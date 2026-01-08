import numpy as np
import math
from typing import List, Tuple, Dict, Any

# ---------------------------
# Activations (unchanged semantics, minor doc fixes)
# ---------------------------
def sigmoid(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = 1.0 / (1.0 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    Z = cache
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

# ---------------------------
# Initialization (He/Xavier option)
# ---------------------------
def initialize_parameters_deep(layer_dims: List[int], init: str = "he", seed: int = None) -> Dict[str, np.ndarray]:
    """
    layer_dims: e.g. [12288, 20, 7, 5, 1]
    init: "he" (recommended for relu), "xavier", or "small"
    """
    if seed is not None:
        np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        fan_in = layer_dims[l-1]
        fan_out = layer_dims[l]
        if init == "he":
            parameters['W' + str(l)] = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        elif init == "xavier":
            parameters['W' + str(l)] = np.random.randn(fan_out, fan_in) * np.sqrt(1.0 / fan_in)
        else:
            parameters['W' + str(l)] = np.random.randn(fan_out, fan_in) * 0.01
        parameters['b' + str(l)] = np.zeros((fan_out, 1))
    return parameters

# ---------------------------
# Forward / Backward helpers
# ---------------------------
def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    else:
        raise ValueError("activation must be 'relu' or 'sigmoid'")
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X: np.ndarray, parameters: Dict[str, np.ndarray]):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL: np.ndarray, Y: np.ndarray) -> float:
    m = Y.shape[1]
    eps = 1e-8
    AL = np.clip(AL, eps, 1 - eps)
    cost = - (1.0 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return float(cost)

def linear_backward(dZ: np.ndarray, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1.0 / m) * np.dot(dZ, A_prev.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA: np.ndarray, cache, activation: str):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError("activation must be 'relu' or 'sigmoid'")
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL: np.ndarray, Y: np.ndarray, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], learning_rate: float) -> Dict[str, np.ndarray]:
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters

def random_mini_batches(X: np.ndarray, Y: np.ndarray, mini_batch_size: int = 64, seed: int = None):
    m = X.shape[1]
    if seed is not None:
        np.random.seed(seed)
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y.reshape(1, m)[:, permutation]
    mini_batches = []
    num_complete = m // mini_batch_size
    for k in range(num_complete):
        start = k * mini_batch_size
        mini_batch_X = shuffled_X[:, start:start + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, start:start + mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    if m % mini_batch_size != 0:
        start = num_complete * mini_batch_size
        mini_batch_X = shuffled_X[:, start:]
        mini_batch_Y = shuffled_Y[:, start:]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    return mini_batches

# ---------------------------
# predict + evaluation
# ---------------------------
def predict(X: np.ndarray, parameters: Dict[str, np.ndarray]) -> np.ndarray:
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5).astype(int)
    return predictions

# ---------------------------
# Full model: trains using mini-batches
# ---------------------------
def L_layer_model_mini_batch(X_input: np.ndarray,
                  Y_input: np.ndarray,
                  layers_dims: List[int],
                  learning_rate: float = 0.0075,
                  num_epochs: int = 1000,
                  mini_batch_size: int = 64,
                  print_cost: bool = True,
                  seed: int = None,
                  init: str = "he") -> Tuple[Dict[str, np.ndarray], List[float]]:
    """
    X_input can be:
      - flattened (n_x, m)
      - images (m, height, width, channels) or (height, width, channels, m)
    Y_input should be shape (1, m) or (m,) etc.
    """
    # --- prepare X, Y into canonical shapes
    X = np.array(X_input)
    Y = np.array(Y_input)
    # if images in shape (m, h, w, c):
    if X.ndim == 4 and X.shape[0] == Y.shape[0]:
        # (m, h, w, c) -> (h*w*c, m)
        m = X.shape[0]
        X = X.reshape(m, -1).T
    elif X.ndim == 4 and X.shape[-1] == Y.shape[0]:
        # assume (h,w,c,m)
        X = X.reshape(-1, X.shape[-1])
    elif X.ndim == 3:
        # ambiguous shape, leave as-is
        pass
    else:
        # assume already (n_x, m)
        pass

    n_x = X.shape[0]
    m = X.shape[1]
    # canonicalize Y
    Y = Y.reshape(1, m)

    # normalization
    if X.max() > 1.0:
        X = X / 255.0

    parameters = initialize_parameters_deep(layers_dims, init=init, seed=seed)

    costs = []
    epoch_seed = seed
    for epoch in range(num_epochs):
        # change seed per epoch so minibatches shuffle differently across epochs
        batch_seed = None if epoch_seed is None else epoch_seed + epoch
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed=batch_seed)
        epoch_cost = 0.0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            AL, caches = L_model_forward(minibatch_X, parameters)
            cost = compute_cost(AL, minibatch_Y)
            epoch_cost += cost * minibatch_X.shape[1]  # accumulate weighted by examples
            grads = L_model_backward(AL, minibatch_Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
        epoch_cost = epoch_cost / m
        costs.append(epoch_cost)
        if print_cost and (epoch % max(1, num_epochs // 10) == 0):
            print(f"Epoch {epoch}/{num_epochs} - cost: {epoch_cost:.6f}")
    return parameters, costs


def inverse_time_decay(lr0, epoch, decay_rate):
    return lr0 / (1 + decay_rate * epoch)

def Mini_batch_with_Learning_Rate_Decay_Model(
    X, Y, layers_dims,
    learning_rate0=0.01,
    num_epochs=1000,
    mini_batch_size=64,
    decay_rate=None,
    decay_fn=None,
    print_cost=True
):
    parameters = initialize_parameters_deep(layers_dims, init="he")
    costs = []

    for epoch in range(num_epochs):

        # 🔽 learning rate decay (ONCE per epoch)
        if decay_fn is not None and decay_rate is not None:
            learning_rate = decay_fn(learning_rate0, epoch, decay_rate)
        else:
            learning_rate = learning_rate0

        minibatches = random_mini_batches(X, Y, mini_batch_size)

        epoch_cost = 0
        for minibatch_X, minibatch_Y in minibatches:
            AL, caches = L_model_forward(minibatch_X, parameters)
            cost = compute_cost(AL, minibatch_Y)
            grads = L_model_backward(AL, minibatch_Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
            epoch_cost += cost * minibatch_X.shape[1]

        epoch_cost /= X.shape[1]
        costs.append(epoch_cost)

        if print_cost and epoch % 100 == 0:
            print(f"Epoch {epoch} | lr={learning_rate:.6f} | cost={epoch_cost:.6f}")

    return parameters, costs
