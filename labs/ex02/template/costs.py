# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w, lossf = "MSE"):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    if (lossf == "MSE"):
        return (1 /(2 * y.shape[0])) * e.T.dot(e)
    elif (lossf == "MAE"):
        return (1 /y.shape[0]) * np.sum(np.abs(e))
    else:
        raise ValueError
