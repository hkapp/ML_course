# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np


def load_data():
    """load data."""
    data = np.loadtxt("dataEx3.csv", delimiter=",", skiprows=1, unpack=True)
    x = data[0]
    y = data[1]
    return x, y


def load_data_from_ex02(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

"""Code from ex02"""

def compute_loss(y, tx, w, lossf = "MSE"):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    if (lossf == "MSE"):
        e = y - tx.dot(w)
        return (1 /(2 * y.shape[0])) * e.T.dot(e)
    elif (lossf == "MAE"):
        e = y - tx.dot(w)
        return (1 /y.shape[0]) * np.sum(np.abs(e))
    elif(lossf == "RMSE"):
        return (np.sqrt(2*compute_loss(y, tx, w, "MSE")))
    else:
        raise ValueError

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]

def grid_search(y, tx, w0, w1, lossf = "MSE"):
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            losses[i,j] = compute_loss(y, tx, [w0[i], w1[j]], lossf)
    return losses
