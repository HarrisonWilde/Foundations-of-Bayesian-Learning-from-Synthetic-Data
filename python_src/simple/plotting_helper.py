import pickle
import os
from datetime import datetime
from plotting import plot_k, plot_gan, plot_alpha


def plot_metric_(df, metric, title, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''
    df = df.melt(['Alpha', 'Laplace Noise Scale'], var_name='Minimised\nDivergence', value_name=metric)
    plot_alpha(df, metric, title, plot_dir, len(df.Alpha.unique()), output_timestamp)


def plot_metric_k_(df, metric, title, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''
    df2 = df.loc[df['Total Number of Samples'] == df['Number of Real Samples']]
    df2.loc[:, 'Number of Real Samples'] = 'Varying'
    df = df.append(df2)
    df = df.melt(['Number of Real Samples', 'Total Number of Samples', 'Laplace Noise Scale'], var_name='Minimised\nDivergence', value_name=metric)
    plot_k(df, metric, title, plot_dir, len(df['Number of Real Samples'].unique()) * 2, output_timestamp)


def plot_metric_gan_(df, metric, title, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''
    df2 = df.loc[df['Total Number of Samples'] == df['Number of Real Samples']]
    df2.loc[:, 'Number of Real Samples'] = 'Varying'
    df = df.append(df2)
    df = df.melt(['Number of Real Samples', 'Total Number of Samples'], var_name='Minimised\nDivergence', value_name=metric)
    plot_gan(df, metric, title, plot_dir, len(df['Number of Real Samples'].unique()) * 2, output_timestamp)


def append_data_frames(directory, exp, prior):
    '''
    Scans outputs directory for all iterations matching the passed path and appends all of the contained dataframes that have matching priors
    '''
    temp = None
    for fp in os.listdir(directory):
        if exp in fp:
            try:
                with open(f'{directory}/{fp}/{prior}', 'rb') as f:
                    if temp is None:
                        temp = pickle.load(f)
                    else:
                        temp = temp.append(pickle.load(f))
            except Exception:
                pass
    return temp.reset_index(drop=True)


def plot_all_k(directory, exp, prior):
    '''
    Helper function to plot all metrics for a full experiment's iterations
    '''
    df = append_data_frames(directory, exp, prior)
    df = df.loc[df['Total Number of Samples'] != 0]
    now = datetime.now().strftime("_%H.%M.%S")
    plot_metric_k_(df.filter(regex='Log Loss|Number of|Laplace Noise'), 'Log Loss', prior, now, 'plots')
    plot_metric_k_(df.filter(regex='KLD|Number of|Laplace Noise'), 'KLD', prior, now, 'plots')
    plot_metric_k_(df.filter(regex='HellingerD|Number of|Laplace Noise'), 'HellingerD', prior, now, 'plots')
    plot_metric_k_(df.filter(regex='TVD|Number of|Laplace Noise'), 'TVD', prior, now, 'plots')
    plot_metric_k_(df.filter(regex='WassersteinD|Number of|Laplace Noise'), 'WassersteinD', prior, now, 'plots')
