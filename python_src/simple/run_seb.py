import argparse
import math
import multiprocessing
import os
import pandas as pd
import numpy as np
from functools import partial
from datetime import datetime
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from setup import apply_noise, generate_data, open_models, generate_ytildes, calculate_dgp_pdf, conditional_get_pairs
from experiment import run_experiment
from evaluation import evaluate_fits
from plotting import plot_metric_k, plot_pdfs
from plotting_helper import plot_all_k


def run_dgp(args, models, dgps, prior_config, window, seed, i):

    proc_num = multiprocessing.current_process()._identity[0]
    a = tqdm(total=0, position=3 + (4 * (proc_num - 1)), bar_format='{desc}', leave=False)
    b = tqdm(total=0, position=5 + (4 * (proc_num - 1)), bar_format='{desc}', leave=False)
    mu, sigma, k_real, k_synth, scale = dgps[i]
    a.set_description_str(f'DGP: y_real[{k_real}] ~ Normal({mu}, {sigma}), y_synth[{k_synth}] ~ Normal({mu}, {sigma}) + Laplace({scale})')

    if args.ytildeconfig is None:
        ytildes = generate_ytildes(mu - sigma * 3, mu + sigma * 3, args.ytildestep)
    else:
        ytildes = generate_ytildes(*args.ytildeconfig)

    real_data = generate_data(mu, sigma, k_real)
    pre_contam_data = generate_data(mu, sigma, k_synth)
    synth_data = apply_noise(pre_contam_data, scale)
    unseen_data = generate_data(mu, sigma, args.num_unseen)
    pdf_ytilde = calculate_dgp_pdf(ytildes, mu, sigma)

    out = [{
        'chain': chain, 'Number of Real Samples': k_real,
        'Total Number of Samples': k_real + k_synth,
        'Laplace Noise Scale': scale, 'Y Tilde Value': ytildes, 'DGP': pdf_ytilde
    } for chain in range(args.chains)]

    for name, model in tqdm(models.items(), leave=False, position=4 + (4 * (proc_num - 1))):

        b.set_description(f'Running MCMC on the {name} model...')

        fit = run_experiment(
            model, args.warmup, args.iters, args.chains,
            real_data, synth_data, unseen_data, ytildes,
            *prior_config, scale, mu, sigma, args.parallel_chains, args.check_hmc_diag, seed
        )

        if fit is None:
            continue

        for chain in tqdm(range(args.chains), position=6 + (4 * (proc_num - 1)), leave=False):

            log_loss, post_pred, KLD, hellingerD, TVD, wassersteinD = evaluate_fits(fit, window, pdf_ytilde, chain)
            out[chain].update({
                f'{name} Log Loss': log_loss,
                f'{name} Posterior Predictive': post_pred,
                f'{name} KLD': KLD,
                f'{name} HellingerD': hellingerD,
                f'{name} TVD': TVD,
                f'{name} WassersteinD': wassersteinD
            })

    if args.plot_pdfs:
        b.set_description(f'Plotting graphs...')
        out = pd.DataFrame(out)
        plot_pdfs(out.filter(regex='Posterior Predictive|Y Tilde Value|DGP'), f"{seed}_{str(k_real)}_{str(k_synth)}", plot_dir)

    return out


def run(args, models, dgps, prior_config, window, iteration):
    '''
    Parent process to run across all combinations of passed parameters, sampling and saving / plotting outputs
    '''
    seed = iteration + args.base_seed
    np.random.seed(seed)
    func = partial(run_dgp, args, models, dgps, prior_config, window, seed)
    outs = process_map(func, range(len(dgps)), max_workers=args.parallel_dgps, leave=False, position=2)
    df = pd.DataFrame([item for sublist in outs for item in sublist])
    df.to_pickle(f'{output_dir}/out_{iteration}.pkl')
    if args.plot_metrics:
        plot_metric_k(df.filter(regex='Log Loss|Number of|Laplace Noise'), 'Log Loss', prior_config, dgp, seed, plot_dir)
        plot_metric_k(df.filter(regex='KLD|Number of|Laplace Noise'), 'KLD', prior_config, dgp, seed, plot_dir)
        plot_metric_k(df.filter(regex='HellingerD|Number of|Laplace Noise'), 'HellingerD', prior_config, dgp, seed, plot_dir)
        plot_metric_k(df.filter(regex='TVD|Number of|Laplace Noise'), 'TVD', prior_config, dgp, seed, plot_dir)
        plot_metric_k(df.filter(regex='WassersteinD|Number of|Laplace Noise'), 'WassersteinD', prior_config, dgp, seed, plot_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments between KL and betaD models.')
    parser.add_argument('-m', '--models', default=None, nargs='+', help='name of KLD stan model (without ".stan")')
    parser.add_argument('-u', '--num_unseen', default=None, type=int, help='amount of unseen samples')
    parser.add_argument('-pm', '--priormu', default=None, type=float, help='normal prior mu to test')
    parser.add_argument('-pa', '--prioralpha', default=None, type=float, help='inverse gamma prior alpha to test')
    parser.add_argument('-pb', '--priorbeta', default=None, type=float, help='inverse gamma prior beta to test')
    parser.add_argument('-hp', '--hyperprior', default=None, type=float, help='hyper prior for variance to test')
    parser.add_argument('-w', '--weight', default=None, type=float, help='w to test for weighted models')
    parser.add_argument('-b', '--beta', default=None, type=float, help='beta to test for beta-divergence')
    parser.add_argument('-bw', '--betaw', default=None, type=float, help='beta w to test for beta-divergence weighted models')
    parser.add_argument('-mu', '--mus', default=None, type=float, nargs='+', help='Space separated list of mus to test')
    parser.add_argument('-s', '--sigmas', default=None, type=float, nargs='+', help='Space separated list of sigmas to test')
    parser.add_argument('-r', '--scales', default=None, type=float, nargs='+', help='Space separated list of noise scale parameters to test')
    parser.add_argument('-wa', '--warmup', default=500, type=int, help='Number of warmup iterations for MCMC')
    parser.add_argument('-i', '--iters', default=5000, type=int, help='Number of recorded iterations for MCMC')
    parser.add_argument('-c', '--chains', default=1, type=int, help='Number of MCMC chains')
    parser.add_argument('-yt', '--ytildeconfig', default=None, type=float, nargs=3, help='Provide ytilde grid config "start end freq" e.g. 0 10 0.1')
    parser.add_argument('-yts', '--ytildestep', default=0.1, type=float, help='Provide ytilde step to use with default range e.g. 0.1')
    parser.add_argument('--iterations', default=1, type=int, help='Number of cpus to use for parallel full iterations, defaults to 1')
    parser.add_argument('--parallel_chains', default=1, type=int, help='Number of cpus to use for parallel chain sampling, defaults to 1')
    parser.add_argument('--parallel_dgps', default=1, type=int, help='Number of cpus to use for parallel dgps, defaults to 1')
    parser.add_argument('-p1', '--plot_pdfs', action='store_true', help='Flag to indicate whether plotting should happen automatically during execution')
    parser.add_argument('-p2', '--plot_metrics', action='store_true', help='Flag to indicate whether plotting should happen automatically during execution')
    parser.add_argument('-mult', '--multiplier', default=1, type=int, help='')
    parser.add_argument('-chd', '--check_hmc_diag', action='store_true', help='Flag to indicate whether to run check_hmc_diagnostics after sampling')
    parser.add_argument('-bs', '--base_seed', default=0, type=int, help='Seed to start counting from')
    args = parser.parse_args()
    print('')
    now = datetime.now().strftime("%m-%d-%Y_%H.%M.%S")

    # Set up all combinations of passed parameters for MCMC and DGP
    models = dict(open_models(args.models, verbose=False))

    n_reals = 10 * np.array(range(1, 11)) ** 2
    n_synths = list(range(1, 10)) + list(range(10, 30, 2)) + list(range(30, 101, 5))
    prior_config = [
        args.priormu,
        args.prioralpha,
        args.priorbeta,
        args.beta,
        args.hyperprior,
        args.weight,
        args.betaw
    ]
    dgps = [(mu, sigma, k_real, k_synth, scale)
        for mu in args.mus
        for sigma in args.sigmas
        for k_real in n_reals
        for k_synth in n_synths
        for scale in args.scales
    ]
    window = args.iters - args.warmup
    print('')

    # Set up directories
    plot_dir = f'plots/sebexp_{now}'
    output_dir = f'outputs/sebexp_{now}'
    paths = ['plots', plot_dir, 'outputs', output_dir]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    if os.path.exists('sampling.txt'):
        os.remove('sampling.txt')

    tqdm(total=0, position=0, bar_format='{desc}').set_description_str(
        f'Running {args.iterations} iterations with {args.chains} chains on {args.parallel_chains} cores')
    func = partial(run, )
    for iteration in tqdm(range(args.iterations), position=1):
        run(args, models, dgps, prior_config, window, iteration)

    for prior_config, dgp in [(prior_config, dgp) for prior_config in prior_configs for dgp in dgps]:
        plot_all_k(path, f'prior{str(prior_config)}dgp{str(dgp)}.pkl')
