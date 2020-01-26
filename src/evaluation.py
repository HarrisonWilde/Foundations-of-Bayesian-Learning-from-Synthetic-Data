import numpy as np
import scipy.stats as st
from numba import njit


@njit
def calculate_log_densities(log_likes):
    return np.log(np.sum(np.exp(log_likes), axis=1)) - np.log(len(log_likes[0]))


@njit
def approximate_posterior_predictive(log_likes_tildes):
    return np.exp(calculate_log_densities(log_likes_tildes))


@njit
def calculate_log_loss(log_likes_unseen):
    return -np.mean(calculate_log_densities(log_likes_unseen))


@njit
def sum_normalised(values):
    return values / values.sum()


@njit
def kl_distance(values1, values2):
    return np.sum(values1 * np.log(values1 / values2))


@njit
def hellinger_distance(values1, values2):
    return np.sqrt(1 - np.dot(np.sqrt(values1), np.sqrt(values2)))


@njit
def tv_distance(values1, values2):
    return np.sum(np.abs(values1 - values2)) / 2


def probability_distances(post_pred, pdf_ytilde):
    '''
    Returns probability distances between a given posterior predictive and the PDF of the DGP
    '''

    post_pred = sum_normalised(post_pred)
    pdf_ytilde = sum_normalised(pdf_ytilde)

    kld = kl_distance(pdf_ytilde, post_pred)
    hellinger = hellinger_distance(pdf_ytilde, post_pred)
    tvd = tv_distance(pdf_ytilde, post_pred)
    wasserstein = st.wasserstein_distance(pdf_ytilde, post_pred)

    return kld, hellinger, tvd, wasserstein


def evaluate_fits(fit, window, pdf_ytilde, chain):
    '''
    Calculates log loss, probability distances and a vector of posterior predictives given the fit outcome for a model
    '''

    indices = [i for i in range(chain * window, (chain + 1) * window)]

    # Calculate log densities at the unseen data values to average to a log loss score
    log_loss = calculate_log_loss((fit['log_likes_unseen'].T)[indices].T)

    # Approximate posterior predictive using ytildes from linspace to use in probability distance calculations
    post_pred = approximate_posterior_predictive((fit['log_likes_tildes'].T)[indices].T)

    # Calculate the KL, Hellinger, TVD and Wasserstein distances between posterior predictive and true pdf
    kld, hellinger, tvd, wasserstein = probability_distances(post_pred, pdf_ytilde)

    return log_loss, post_pred, kld, hellinger, tvd, wasserstein
