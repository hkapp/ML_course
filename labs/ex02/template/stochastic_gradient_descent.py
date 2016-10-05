# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for MSE."""
    N = len(y)
    idx = np.random.randint(0, N)
    g = tx[idx] * (tx[idx].dot(w) - y[idx])
    return g


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm for MSE."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_epochs):
        rgrads = [compute_stoch_gradient(y, tx, w) for i in range(batch_size)]
        gradient = np.mean(rgrads, axis = 0)
        loss = compute_loss(y, tx, w, "MSE")
        w = w - gamma * gradient
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
