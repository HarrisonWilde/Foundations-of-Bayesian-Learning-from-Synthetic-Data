import pystan
import pandas as pd
import pickle
import os
import pategan


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


def load_data(original_path, eps, gan, targets, sep, num_teachers, epochs, delta, split, b):

    path = f'{original_path.replace("raw", "splits")}_{"".join(targets)}_split{split}_gan{gan}'

    if (os.path.exists(
        f'{path}_train.csv') and os.path.exists(
        f'{path}_test.csv') and os.path.exists(
        f'{path}_eps{str(eps)}_train.csv') and os.path.exists(
        f'{path}_eps{str(eps)}_test.csv')):

        b.set_description_str(f'Loaded {path} from previous runs')

        train = pd.read_csv(f'{path}_train.csv')
        test = pd.read_csv(f'{path}_test.csv')
        synth_train = pd.read_csv(f'{path}_eps{str(eps)}_train.csv')
        synth_test = pd.read_csv(f'{path}_eps{str(eps)}_test.csv')

    else:
        if gan == 'pate':
            train, test, synth_train, synth_test = pategan.run(original_path, path, targets, sep, num_teachers, epochs, eps, delta, split, b)

    return train, test, synth_train, synth_test


def conditional_get_pairs(real_alphas, contam_alphas, max_sum):
    return [(a1, a2) for a1 in real_alphas for a2 in contam_alphas if a1 + a2 <= max_sum + 1e-6]
