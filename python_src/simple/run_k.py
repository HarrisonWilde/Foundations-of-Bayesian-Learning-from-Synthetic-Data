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


def run(args, now, models, dgps, num_alphas, prior_configs, alpha_configs, window, iteration):
    '''
    Parent process to run across all combinations of passed parameters, sampling and saving / plotting outputs
    '''

    # Set up directories
    plot_dir = f'plots/kexp_{iteration}_{now}'
    output_dir = f'outputs/kexp_{iteration}_{now}'
    paths = ['plots', plot_dir, 'outputs', output_dir]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    if os.path.exists('sampling.txt'):
        os.remove('sampling.txt')

    a = tqdm(total=0, position=2, bar_format='{desc}')
    b = tqdm(total=0, position=4, bar_format='{desc}')
    c = tqdm(total=0, position=6, bar_format='{desc}')
    d = tqdm(total=0, position=8, bar_format='{desc}')
    e = tqdm(total=0, position=10, bar_format='{desc}')
    for dgp in tqdm(dgps, leave=False, position=3):

        seed = iteration + args.base_seed
        np.random.seed(seed)

        a.set_description_str('DGP: y[{2}] ~ Normal({0}, {1})'.format(*dgp))

        if args.ytildeconfig is None:
            ytildes = generate_ytildes(dgp[0] - dgp[1] * 3, dgp[0] + dgp[1] * 3, args.ytildestep)
        else:
            ytildes = generate_ytildes(*args.ytildeconfig)
        data = generate_data(*dgp)
        pre_contam_data = generate_data(*dgp)
        unseen_data = generate_data(*dgp)
        pdf_ytilde = calculate_dgp_pdf(ytildes, *dgp)

        for prior_config in tqdm(prior_configs, leave=False, position=5):

            b.set_description_str('Prior: sigma2 ~ InverseGamma({1}, {2}), mu ~ Normal({0}, {4} * sigma2), Beta: {3}, W: {5}, Beta W: {6}'
                                  .format(*prior_config))
            output_timestamp = datetime.now().strftime("_%H.%M.%S")
            outs = []

            for scale in tqdm(args.scales, leave=False, position=7):

                c.set_description_str(f'Laplace Noise Scale: {scale}')

                contam_data, _ = apply_noise(pre_contam_data, scale, 0)

                for real_alpha, contam_alpha in tqdm(alpha_configs, leave=False, position=9):

                    d.set_description_str(f'Number of Real Samples: {math.floor(real_alpha*len(data))}, Number of Contaminated Samples: {math.floor(contam_alpha*len(contam_data))}')

                    out = [{'chain': chain, 'Number of Real Samples': math.floor(real_alpha * len(data)),
                            'Total Number of Samples': math.floor(real_alpha * len(data)) + math.floor(contam_alpha * len(contam_data)),
                            'Laplace Noise Scale': scale, 'Y Tilde Value': ytildes, 'DGP': pdf_ytilde}
                           for chain in range(args.chains)]

                    for name, model in tqdm(models.items(), leave=False, position=11):

                        e.set_description(f'Running MCMC on the {name} model...')

                        fit = run_experiment(model, args.warmup, args.iters, args.chains, data[:math.floor(real_alpha * len(data))],
                                             contam_data[:math.floor(contam_alpha * len(contam_data))], unseen_data, ytildes,
                                             *prior_config, scale, *dgp, 0, args.check_hmc_diag, seed)

                        if fit is None:
                            continue

                        for chain in tqdm(range(args.chains), position=12, leave=False):

                            log_loss, post_pred, KLD, hellingerD, TVD, wassersteinD = evaluate_fits(fit, window, pdf_ytilde, chain)
                            out[chain].update({f'{name} Log Loss': log_loss, f'{name} Posterior Predictive': post_pred, f'{name} KLD': KLD,
                                               f'{name} HellingerD': hellingerD, f'{name} TVD': TVD, f'{name} WassersteinD': wassersteinD})

                    outs.extend(out)

                    # MultiProcessing code, doesn't seem to work as it should at the moment

                    # chain_kl_log_losses, chain_beta_log_losses, chain_prob_distances = list(zip(*p_umap(
                    #     evaluate_fits, klfit, betafit, window, output_name, plot_dir, ytildes, pdf_ytilde, list(range(args.chains)), num_cpus=args.cpu_count, leave=False)))
                    # kl_log_losses.extend(chain_kl_log_losses)
                    # beta_log_losses.extend(chain_beta_log_losses)
                    # prob_distances.extend(chain_prob_distances)

                    if args.plot_pdfs:
                        out = pd.DataFrame(out)
                        plot_pdfs(out.filter(regex='Posterior Predictive|Y Tilde Value|DGP'), output_timestamp + '_' + str(real_alpha) + '_' + str(contam_alpha), plot_dir)

            df = pd.DataFrame(outs)
            df.to_pickle(f'{output_dir}/prior{str(prior_config)}dgp{str(dgp)}.pkl')
            if args.plot_metrics:
                plot_metric_k(df.filter(regex='Log Loss|Number of|Laplace Noise'), 'Log Loss', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric_k(df.filter(regex='KLD|Number of|Laplace Noise'), 'KLD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric_k(df.filter(regex='HellingerD|Number of|Laplace Noise'), 'HellingerD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric_k(df.filter(regex='TVD|Number of|Laplace Noise'), 'TVD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric_k(df.filter(regex='WassersteinD|Number of|Laplace Noise'), 'WassersteinD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments between KL and betaD models.')
    parser.add_argument('-m', '--models', default=None, nargs='+', help='Name of KLD stan model (without ".stan")')
    parser.add_argument('-ra', '--real_alphas', default=None, type=float, nargs='+', help='Space separated list of alpha splits to test')
    parser.add_argument('-ca', '--contaminated_alphas', default=None, type=float, nargs='+', help='Space separated list of alpha splits to test')
    parser.add_argument('-k', '--ks', default=None, type=int, nargs='+', help='Space separated list of ks (no. DGP sampled points) to test')
    parser.add_argument('-pm', '--priormus', default=None, type=float, nargs='+', help='Space separated list of normal prior mus to test')
    parser.add_argument('-pa', '--prioralphas', default=None, type=float, nargs='+', help='Space separated list of inverse gamma prior alphas to test')
    parser.add_argument('-pb', '--priorbetas', default=None, type=float, nargs='+', help='Space separated list of inverse gamma prior betas to test')
    parser.add_argument('-hp', '--hyperpriors', default=None, type=float, nargs='+', help='Space separated list of hyper priors for variance to test')
    parser.add_argument('-b', '--betas', default=None, type=float, nargs='+', help='Space separated list of betas to test for beta-divergence')
    parser.add_argument('-w', '--ws', default=None, type=float, nargs='+', help='Space separated list of ws to test for weighted models')
    parser.add_argument('-bw', '--betaws', default=None, type=float, nargs='+', help='Space separated list of beta ws to test for beta-divergence weighted models')
    parser.add_argument('-mu', '--mus', default=None, type=float, nargs='+', help='Space separated list of mus to test')
    parser.add_argument('-s', '--sigmas', default=None, type=float, nargs='+', help='Space separated list of sigmas to test')
    parser.add_argument('-r', '--scales', default=None, type=float, nargs='+', help='Space separated list of noise scale parameters to test')
    parser.add_argument('-wa', '--warmup', default=500, type=int, help='Number of warmup iterations for MCMC')
    parser.add_argument('-i', '--iters', default=5000, type=int, help='Number of recorded iterations for MCMC')
    parser.add_argument('-c', '--chains', default=1, type=int, help='Number of MCMC chains')
    parser.add_argument('-yt', '--ytildeconfig', default=None, type=float, nargs=3, help='Provide ytilde grid config "start end freq" e.g. 0 10 0.1')
    parser.add_argument('-yts', '--ytildestep', default=0.1, type=float, help='Provide ytilde step to use with default range e.g. 0.1')
    parser.add_argument('-cpu', '--cpu_count', default=multiprocessing.cpu_count(), type=int, help='Number of cpus to use, defaults to maximum available')
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
    dgps = [(mu, sigma, k) for mu in args.mus for sigma in args.sigmas for k in args.ks]
    num_alphas = len(args.real_alphas) + len(args.contaminated_alphas)
    prior_configs = [(priormu, priora, priorb, beta, hyperprior, w, beta_w)
                     for priormu in args.priormus for priora in args.prioralphas for priorb in args.priorbetas for beta in args.betas
                     for hyperprior in args.hyperpriors for w in args.ws for beta_w in args.betaws]
    alpha_configs = conditional_get_pairs(args.real_alphas, args.contaminated_alphas, 1)
    window = args.iters - args.warmup
    print('')

    tqdm(total=0, position=0, bar_format='{desc}').set_description_str(
        f'Running {args.cpu_count * args.multiplier} iterations with {args.chains} chains each on {args.cpu_count} cores')
    func = partial(run, args, now, models, dgps, num_alphas, prior_configs, alpha_configs, window)
    process_map(func, range(args.cpu_count * args.multiplier), max_workers=args.cpu_count, leave=False, position=1)

    for prior_config, dgp in [(prior_config, dgp) for prior_config in prior_configs for dgp in dgps]:
        plot_all_k(path, f'prior{str(prior_config)}dgp{str(dgp)}.pkl')