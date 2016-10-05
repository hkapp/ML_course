# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs


def compute_gradient(y, tx, w, lossf = "MSE"):
    c = -1 / len(y)
    e = y - tx.dot(w)
    if (lossf == "MSE"):
        return c * tx.T.dot(e)
    elif (lossf == "MAE"):
        return c * tx.T.dot(np.sign(e))
    else:
        raise ValueError


def gradient_descent(y, tx, initial_w, max_iters, gamma, lossf = "MSE"):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w, lossf)
        print(gradient)
        loss = compute_loss(y, tx, w, lossf)
        w = w - gamma * gradient
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
