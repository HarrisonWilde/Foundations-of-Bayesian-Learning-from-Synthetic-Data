import argparse
import math
import multiprocessing
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
from setup import load_data, open_models, conditional_get_pairs
from experiment import run_experiment
from evaluation import evaluate_fits
from plotting import plot_metric


def run(args, now, models, prior_configs, window, train, test, synth_train, synth_test, features, seed):
    '''
    Parent process to run across all combinations of passed parameters, sampling and saving / plotting outputs
    '''

    # Set up directories
    plot_dir = f'plots/exp_{seed}_{now}'
    output_dir = f'outputs/exp_{seed}_{now}'
    paths = ['plots', plot_dir, 'outputs', output_dir]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    if os.path.exists('sampling.txt'):
        os.remove('sampling.txt')

    c = tqdm(total=0, position=6, bar_format='{desc}')
    d = tqdm(total=0, position=8, bar_format='{desc}')
    e = tqdm(total=0, position=10, bar_format='{desc}')

    for prior_config in tqdm(prior_configs, leave=False, position=7):

        c.set_description_str('Prior: Weight = {0}, Beta = {1}, Beta Weight = {2}'.format(*prior_config))
        output_timestamp = datetime.now().strftime("_%H.%M.%S")
        outs = []

        for real_alpha, synth_alpha in tqdm(conditional_get_pairs(args.real_alphas, args.synth_alphas, 1), leave=False, position=9):

            d.set_description_str(
                f'Number of Real Samples: {math.floor(real_alpha * len(train))}, Number of Contaminated Samples: {math.floor(synth_alpha * len(synth_train))}')

            out = [{'chain': chain, 'Number of Real Samples': math.floor(real_alpha * len(train)),
                    'Total Number of Samples': math.floor(real_alpha * len(train)) + math.floor(synth_alpha * len(synth_train))}
                   for chain in range(args.chains)]

            for name, model in tqdm(models.items(), leave=False, position=11):

                e.set_description(f'Running MCMC on the {name} model...')

                data = dict(
                    f=len(features),
                    a=math.floor(real_alpha * len(train)),
                    X_real=train.iloc[0:math.floor(real_alpha * len(train))].loc[:, features].values,
                    y_real=train.iloc[0:math.floor(real_alpha * len(train))].loc[:, args.targets].values.flatten(),
                    b=math.floor(synth_alpha * len(synth_train)),
                    X_synth=train.iloc[0:math.floor(synth_alpha * len(synth_train))].loc[:, features].values,
                    y_synth=train.iloc[0:math.floor(synth_alpha * len(synth_train))].loc[:, args.targets].values.flatten(),
                    c=len(test),
                    X_test=test[features].values,
                    y_test=test[args.targets].values.flatten(),
                    w=prior_config[0],
                    beta=prior_config[1],
                    beta_w=prior_config[2])

                fit = run_experiment(model, data, args.warmup, args.iters, args.chains, args.n_jobs, args.check_hmc_diag, seed)

                if fit is None:
                    continue

                for chain in tqdm(range(args.chains), position=12, leave=False):

                    log_loss, avg_roc_auc = evaluate_fits(fit, test[args.targets].values.flatten(), window, chain)
                    out[chain].update({f'{name} Log Loss': log_loss, f'{name} ROC AUC': avg_roc_auc})

                outs.extend(out)
                print(out)
                df = pd.DataFrame(outs)
                df.to_pickle(f'{output_dir}/conf{str(prior_config)}.pkl')

        if args.plot:
            plot_metric(df.filter(regex='Log Loss|Number of'), 'Log Loss', prior_config, output_timestamp, plot_dir)
            plot_metric(df.filter(regex='ROC AUC|Number of'), 'ROC AUC', prior_config, output_timestamp, plot_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments between KL and betaD models')
    parser.add_argument('-m', '--models', default=None, nargs='+', help='Name of KLD stan model (without ".stan")')
    parser.add_argument('-d', '--data', default='data/raw/creditcard', help='Path of data to use')
    parser.add_argument('-e', '--epsilons', default=1, type=float, nargs='+', help='Epsilon guarantees to test for contaminated data')
    parser.add_argument('-ra', '--real_alphas', default=None, type=float, nargs='+', help='Space separated list of alpha splits to test')
    parser.add_argument('-sa', '--synth_alphas', default=None, type=float, nargs='+', help='Space separated list of alpha splits to test')
    parser.add_argument('-g', '--gan', default='pate', help='Type of GAN to use to generate synthetic data')
    parser.add_argument('-wa', '--warmup', default=500, type=int, help='Number of warmup iterations for MCMC')
    parser.add_argument('-i', '--iters', default=5000, type=int, help='Number of recorded iterations for MCMC')
    parser.add_argument('-c', '--chains', default=1, type=int, help='Number of MCMC chains')
    parser.add_argument('-b', '--betas', default=None, type=float, nargs='+', help='Space separated list of betas to test for beta-divergence')
    parser.add_argument('-w', '--ws', default=None, type=float, nargs='+', help='Space separated list of ws to test for weighted models')
    parser.add_argument('-bw', '--betaws', default=None, type=float, nargs='+', help='Space separated list of beta ws to test for beta-divergence weighted models')
    parser.add_argument('-cpu', '--n_jobs', default=math.floor(multiprocessing.cpu_count() / 2), type=int, help='Number of cpus to use, defaults to maximum available')
    parser.add_argument('-mu', '--multiplier', default=1, type=int, help='Multiplier for running experiments in parallel sequentially')
    parser.add_argument('-p', '--plot', action='store_true', help='Flag to indicate whether plotting should happen automatically during execution')
    parser.add_argument('-chd', '--check_hmc_diag', action='store_true', help='Flag to indicate whether to run check_hmc_diagnostics after sampling')
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs to train the GAN for")
    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--teachers", type=int, default=5)
    parser.add_argument("--targets", nargs='+', help="Name of response var when using csv as input.")
    parser.add_argument("--separator", default=',', help="Separator for the input csv file.")
    parser.add_argument("-tts", "--split", type=float, default=0.7)
    args = parser.parse_args()

    # Set up experiment and all combinations of passed parameters for MCMC
    print('')
    now = datetime.now().strftime("%m-%d-%Y_%H.%M.%S")
    models = dict(open_models(args.models, verbose=False))
    prior_configs = [(w, beta, beta_w) for w in args.ws for beta in args.betas for beta_w in args.betaws]
    window = args.iters - args.warmup
    print('')

    a = tqdm(total=0, position=0, bar_format='{desc}')
    b = tqdm(total=0, position=2, bar_format='{desc}')

    for eps in tqdm(args.epsilons, position=1, leave=False):

        a.set_description_str(f'Dataset: {args.data.split("/")[-1]}, Epsilon: {str(eps)}')

        train, test, synth_train, synth_test = load_data(args.data, eps, args.gan, args.targets, args.separator, args.teachers, args.epochs, args.delta, args.split, b)
        features = list(train.columns)
        for target in args.targets:
            features.remove(target)

        tqdm(total=0, position=4, bar_format='{desc}').set_description_str(
            f'Running {args.n_jobs * args.multiplier} iterations with {args.chains} chain(s) each on {args.n_jobs} cores')
        func = partial(run, args, now, models, prior_configs, window, train, test, synth_train, synth_test, features)
        process_map(func, range(args.n_jobs * args.multiplier), max_workers=args.n_jobs, leave=False, position=5)
