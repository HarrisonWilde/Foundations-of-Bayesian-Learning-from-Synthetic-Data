import argparse
import multiprocessing
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
# from p_tqdm import p_umap
from setup import apply_noise, generate_data, open_models, generate_ytildes, calculate_dgp_pdf
from experiment import run_experiment
from evaluation import evaluate_fits
from plotting import plot_metric, plot_pdfs


def run(args):
    '''
    Parent process to run across all combinations of passed parameters, sampling and saving / plotting outputs
    '''

    # Set up directories
    print('')
    plot_dir = 'plots/exp_' + datetime.now().strftime("%m-%d-%Y_%H.%M.%S")
    output_dir = 'outputs/exp_' + datetime.now().strftime("%m-%d-%Y_%H.%M.%S")
    paths = ['plots', plot_dir, 'outputs', output_dir]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    if os.path.exists('sampling.txt'):
        os.remove('sampling.txt')

    print(f'Using {args.cpu_count} cores')
    os.environ["STAN_NUM_THREADS"] = str(args.cpu_count)

    models = dict(open_models(args.models, verbose=False))

    # Set up all combinations of passed parameters for MCMC and DGP
    dgps = [(mu, sigma, k) for mu in args.mus for sigma in args.sigmas for k in args.ks]
    noise_configs = [(scale, alpha) for scale in args.scales for alpha in args.alphas]
    num_alphas = len(args.alphas)
    prior_configs = [(priormu, priora, priorb, beta, hyperprior, w, beta_w)
                     for priormu in args.priormus for priora in args.prioralphas for priorb in args.priorbetas
                     for beta in args.betas for hyperprior in args.hyperpriors for w in args.ws for beta_w in args.betaws]

    if args.ytildeconfig is not None:
        ytildes = generate_ytildes(*args.ytildeconfig)

    window = args.iters - args.warmup

    a = tqdm(total=0, position=1, bar_format='{desc}')
    b = tqdm(total=0, position=3, bar_format='{desc}')
    c = tqdm(total=0, position=5, bar_format='{desc}')
    d = tqdm(total=0, position=7, bar_format='{desc}')
    print('')

    for dgp in tqdm(dgps, position=0):

        a.set_description_str('DGP: y[{2}] ~ Normal({0}, {1})'.format(*dgp))

        if args.ytildeconfig is None:
            ytildes = generate_ytildes(dgp[0] - dgp[1] * 3, dgp[0] + dgp[1] * 3, args.ytildestep)
        data = generate_data(*dgp)
        unseen_data = generate_data(*dgp)
        pdf_ytilde = calculate_dgp_pdf(ytildes, *dgp)

        for prior_config in tqdm(prior_configs, leave=False, position=2):

            b.set_description_str('Prior: sigma2 ~ InverseGamma({1}, {2}), mu ~ Normal({0}, {4} * sigma2), Beta: {3}, W: {5}, Beta W: {6}'.format(*prior_config))
            output_timestamp = datetime.now().strftime("_%H.%M.%S")
            outs = []

            for noise_config in tqdm(noise_configs, leave=False, position=4):

                c.set_description_str('Laplace Noise Scale = {}, Alpha = {}'.format(*noise_config))

                contaminated_data, indices = apply_noise(data, *noise_config)
                out = [{'chain': chain, 'Alpha': noise_config[1], 'Laplace Noise Scale': noise_config[0], 'Y Tilde Value': ytildes, 'DGP': pdf_ytilde}
                       for chain in range(args.chains)]

                for name, model in tqdm(models.items(), leave=False, position=6):

                    d.set_description(f'Running MCMC on the {name} model...')

                    fit = run_experiment(model, args.warmup, args.iters, args.chains, data[~indices], data[indices],
                                         unseen_data, ytildes, *prior_config, noise_config[0], *dgp, args.cpu_count)

                    if fit is None:
                        continue

                    for chain in tqdm(range(args.chains), position=8, leave=False):

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
                    plot_pdfs(out.filter(regex='Posterior Predictive|Y Tilde Value|DGP'), output_timestamp + '_' + str(noise_config[0]) + '_' + str(noise_config[1]), plot_dir)

            df = pd.DataFrame(outs)
            df.to_pickle(f'{output_dir}/prior{str(prior_config)}dgp{str(dgp)}.pkl')
            if args.plot_metrics:
                plot_metric(df.filter(regex='Log Loss|Alpha|Laplace Noise'), 'Log Loss', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric(df.filter(regex='KLD|Alpha|Laplace Noise'), 'KLD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric(df.filter(regex='HellingerD|Alpha|Laplace Noise'), 'HellingerD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric(df.filter(regex='TVD|Alpha|Laplace Noise'), 'TVD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)
                plot_metric(df.filter(regex='WassersteinD|Alpha|Laplace Noise'), 'WassersteinD', num_alphas, prior_config, dgp, output_timestamp, plot_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments between KL and betaD models.')
    parser.add_argument('-m', '--models', default=None, nargs='+', help='Name of KLD stan model (without ".stan")')
    parser.add_argument('-a', '--alphas', default=None, type=float, nargs='+', help='Space separated list of alpha splits to test')
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
    parser.add_argument('-p1', '--plot_metrics', action='store_true', help='Flag to indicate whether plotting should happen automatically during execution')
    parser.add_argument('-p2', '--plot_pdfs', action='store_true', help='Flag to indicate whether plotting should happen automatically during execution')
    parser.add_argument('-chd', '--check_hmc_diag', action='store_true', help='Flag to indicate whether to run check_hmc_diagnostics after sampling')
    args = parser.parse_args()
    run(args)
