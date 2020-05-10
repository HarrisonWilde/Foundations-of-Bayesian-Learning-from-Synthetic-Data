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


def run_dgp(args, dgp, models, ks, prior_config, all_real_data, all_synth_data, unseen_data, window, plot_dir, seed, i):

    mu, sigma, scale = dgp
    k_real, k_synth = ks[i]
    if args.showprocessing:
        proc_num = multiprocessing.current_process()._identity[0]
        a = tqdm(total=0, position=3 + (4 * (proc_num - 1)), bar_format='{desc}', leave=False)
        b = tqdm(total=0, position=5 + (4 * (proc_num - 1)), bar_format='{desc}', leave=False)
        a.set_description_str(f'DGP: y_real[{k_real}] ~ Normal({mu}, {sigma}), y_synth[{k_synth}] ~ Normal({mu}, {sigma}) + Laplace({scale})')

    real_data = all_real_data[:k_real]
    synth_data = all_synth_data[:k_synth]

    out = [{
        'chain': chain, 'Number of Real Samples': k_real,
        'Total Number of Samples': k_real + k_synth,
        'Laplace Noise Scale': scale, 'Y Tilde Value': ytildes, 'DGP': pdf_ytilde
    } for chain in range(args.chains)]

    if args.showprocessing:
        p1 = tqdm(models.items(), leave=False, position=4 + (4 * (proc_num - 1)))
    else:
        p1 = models.items()

    for name, model in p1:

        if args.showprocessing:
            b.set_description(f'Running MCMC on the {name} model...')

        fit = run_experiment(
            model, args.warmup, args.iters, args.chains,
            real_data, synth_data, unseen_data, ytildes,
            *prior_config, scale, mu, sigma, args.parallel_chains, args.check_hmc_diag, seed
        )

        if fit is None:
            continue

        if args.showprocessing:
            p2 = tqdm(range(args.chains), position=6 + (4 * (proc_num - 1)), leave=False)
        else:
            p2 = range(args.chains)

        for chain in p2:

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
        out_df = pd.DataFrame(out)
        plot_pdfs(out_df.filter(regex='Posterior Predictive|Y Tilde Value|DGP'), f"{seed}_{str(k_real)}_{str(k_synth)}", plot_dir)
    return out


def run(args, models, dgp, prior_config, n_reals, n_synths, ytildes, pdf_ytilde, window, plot_dir, output_dir, iteration):
    '''
    Parent process to run across all combinations of passed parameters, sampling and saving / plotting outputs
    '''
    seed = iteration + args.base_seed
    np.random.seed(seed)
    mu, sigma, scale = dgp
    proportions = [(k_real, k_synth)
        for k_real in n_reals
        for k_synth in n_synths
        if k_real + k_synth <= max(n_reals)
    ]
    
    all_real_data = generate_data(mu, sigma, max(n_reals))
    np.save(f'{output_dir}/real_{iteration}', all_real_data)
    pre_contam_data = generate_data(mu, sigma, max(n_synths))
    all_synth_data = apply_noise(pre_contam_data, scale)
    np.save(f'{output_dir}/synth_{iteration}', all_synth_data)
    unseen_data = generate_data(mu, sigma, args.num_unseen)
    np.save(f'{output_dir}/unseen_{iteration}', unseen_data)

    func = partial(
        run_dgp, args, dgp, models, proportions, prior_config,
        all_real_data, all_synth_data, unseen_data, window, plot_dir, seed
    )
    outs = process_map(func, range(len(proportions)), max_workers=args.parallel_dgps, leave=False, position=2)
    df = pd.DataFrame([item for sublist in outs for item in sublist])
    df.to_pickle(f'{output_dir}/out_{iteration}.pkl')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments between KL and betaD models.')
    parser.add_argument('-m', '--models', default=None, nargs='+', help='name of KLD stan model (without ".stan")')
    parser.add_argument('-od', '--outdir', default="plots", help='Output directory path')
    parser.add_argument('-pd', '--plotdir', default="outputs", help='Output directory path for plots')
    parser.add_argument('-u', '--num_unseen', default=None, type=int, help='amount of unseen samples')
    parser.add_argument('-pm', '--priormu', default=None, type=float, help='normal prior mu to test')
    parser.add_argument('-pa', '--prioralpha', default=None, type=float, help='inverse gamma prior alpha to test')
    parser.add_argument('-pb', '--priorbeta', default=None, type=float, help='inverse gamma prior beta to test')
    parser.add_argument('-hp', '--hyperprior', default=None, type=float, help='hyper prior for variance to test')
    parser.add_argument('-w', '--weight', default=None, type=float, help='w to test for weighted models')
    parser.add_argument('-b', '--beta', default=None, type=float, help='beta to test for beta-divergence')
    parser.add_argument('-bw', '--betaw', default=None, type=float, help='beta w to test for beta-divergence weighted models')
    parser.add_argument('-mu', '--mu', default=None, type=float, help='Space separated list of mus to test')
    parser.add_argument('-s', '--sigma', default=None, type=float, help='Space separated list of sigmas to test')
    parser.add_argument('-r', '--scale', default=None, type=float, help='Space separated list of noise scale parameters to test')
    parser.add_argument('-wa', '--warmup', default=500, type=int, help='Number of warmup iterations for MCMC')
    parser.add_argument('-i', '--iters', default=5000, type=int, help='Number of recorded iterations for MCMC')
    parser.add_argument('-c', '--chains', default=1, type=int, help='Number of MCMC chains')
    parser.add_argument('-yt', '--ytildeconfig', default=None, type=float, nargs=3, help='Provide ytilde grid config "start end freq" e.g. 0 10 0.1')
    parser.add_argument('-yts', '--ytildestep', default=0.1, type=float, help='Provide ytilde step to use with default range e.g. 0.1')
    parser.add_argument('--iterations', default=1, type=int, help='Number of full iterations, defaults to 1')
    parser.add_argument('--parallel_chains', default=1, type=int, help='Number of cpus to use for parallel chain sampling, defaults to 1')
    parser.add_argument('--parallel_dgps', default=1, type=int, help='Number of cpus to use for parallel dgps, defaults to 1')
    parser.add_argument('-p1', '--plot_pdfs', action='store_true', help='Flag to indicate whether plotting should happen automatically during execution')
    parser.add_argument('-p2', '--plot_metrics', action='store_true', help='Flag to indicate whether plotting should happen automatically during execution')
    parser.add_argument('-mult', '--multiplier', default=1, type=int, help='')
    parser.add_argument('-chd', '--check_hmc_diag', action='store_true', help='Flag to indicate whether to run check_hmc_diagnostics after sampling')
    parser.add_argument('-sp', '--showprocessing', action='store_true', help='Flag to indicate whether to show progress bars for parallel dgps (keep off on cluster)')
    parser.add_argument('-bs', '--base_seed', default=0, type=int, help='Seed to start counting from')
    args = parser.parse_args()
    print('')
    now = datetime.now().strftime("%m-%d-%Y_%H.%M.%S")

    # Set up all combinations of passed parameters for MCMC and DGP
    models = dict(open_models(args.models, verbose=False))

    n_reals = [1, 5] + list(10 * np.array(range(1, 11)) ** 2)
    n_synths = list(range(0, 30, 2)) + list(range(30, 150, 5)) + list(range(150, 1001, 100))
    prior_config = [
        args.priormu,
        args.prioralpha,
        args.priorbeta,
        args.beta,
        args.hyperprior,
        args.weight,
        args.betaw
    ]
    dgp = [
        args.mu,
        args.sigma,
        args.scale
    ]
    mu, sigma, scale = dgp
    if args.ytildeconfig is None:
        ytildes = generate_ytildes(mu - sigma * 3, mu + sigma * 3, args.ytildestep)
    else:
        ytildes = generate_ytildes(*args.ytildeconfig)
    pdf_ytilde = calculate_dgp_pdf(ytildes, mu, sigma)

    window = args.iters - args.warmup
    print('')

    # Set up directories
    plot_dir = f'{args.plotdir}/sebexp_{now}'
    output_dir = f'{args.outdir}/sebexp_{now}'
    paths = [args.plotdir, plot_dir, args.outdir, output_dir]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    if os.path.exists('sampling.txt'):
        os.remove('sampling.txt')

    tqdm(total=0, position=0, bar_format='{desc}').set_description_str(
        f'Running {args.iterations} iterations with {args.chains} chains on {args.parallel_chains} cores')
    func = partial(run, )
    for iteration in tqdm(range(args.iterations), position=1):
        run(args, models, dgp, prior_config, n_reals, n_synths, ytildes, pdf_ytilde, window, plot_dir, output_dir, iteration)
