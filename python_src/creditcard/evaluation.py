import numpy as np
from sklearn.metrics import roc_auc_score
from numba import njit


@njit
def calculate_log_densities(log_likes):
    return np.log(np.sum(np.exp(log_likes), axis=0)) - np.log(len(log_likes))


@njit
def calculate_log_loss(log_likes_test, mutli=True):
    return -np.mean(calculate_log_densities(log_likes_test))


def evaluate_fits(fit, y_test, window, chain):
    """
    Calculates log loss, probability distances and a vector of posterior predictives given the fit outcome for a model
    """
    indices = [i for i in range(chain * window, (chain + 1) * window)]

    # # Calculate log densities at the unseen test data values to average to a log loss score
    log_loss_single = -calculate_log_densities(fit['log_like_test'][indices])
    log_loss = calculate_log_loss(fit['log_likes_test'][indices])
    avg_roc_auc = roc_auc_score(y_test, np.mean(fit['probabilities_test'][indices], axis=0))

    return log_loss_single, log_loss, avg_roc_auc
