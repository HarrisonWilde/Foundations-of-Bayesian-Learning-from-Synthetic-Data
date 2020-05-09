import pystan
import numpy as np
import math
import pickle
import os
import scipy.stats as st


def StanModel_cache(filename, verbose, **kwargs):
    """
    Use just as you would build
    """
    cache_fn = f'models/cached-model-{filename}.pkl'
    with open(f'models/{filename}.stan') as f:
        name = f.readline()[3:-1]
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except Exception:
        sm = pystan.StanModel(file=f'models/{filename}.stan', verbose=verbose)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print(f"Succesfully loaded the {name} model from cache.")
    return (name, sm)


def open_models(filenames, verbose=True, gan=False):
    '''
    Returns generator of tuples of a files name scraped from its first line and a string containing the content of the file
    '''
    if filenames is not None:
        for filename in filenames:
            if os.path.exists(f'models/{filename}.stan'):
                yield StanModel_cache(filename, verbose)
            else:
                print(f"{filename} cannot be found...")
    else:
        for filename in os.listdir('models'):
            if filename.rsplit('.', 1)[-1] == 'stan':
                yield StanModel_cache(filename[:-5], verbose)


def generate_data(mu, sigma, k):
    '''
    Generates k normal samples from Norm(mu, sigma)
    '''
    return np.random.normal(mu, sigma, k)


def apply_noise(data, scale):
    '''
    Applies Laplace(0, scale) noise to the data
    '''
    indices = np.ones(len(data))
    return data + np.random.laplace(0, scale, len(data))


def generate_ytildes(start, finish, step):
    '''
    Generates a linnspace / grid
    '''
    return np.arange(start, finish, step)


def calculate_dgp_pdf(ytilde, mu, sigma):
    '''
    Returns values of the dgp pdf at all passed ytilde
    '''
    return np.array([st.norm(mu, sigma).pdf(x) for x in ytilde])


def conditional_get_pairs(real_alphas, contam_alphas, max_sum):
    return [(a1, a2) for a1 in real_alphas for a2 in contam_alphas if a1 + a2 <= max_sum + 1e-6]
