import pickle
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(df, y, plot_dir, num_alphas):
    '''
    Generic plot function for plotting number of samples / contaminated samples on x-axis and metric on y-axis
    '''
    # g = sns.catplot(x='Alpha', y=y, hue='Minimised\nDivergence', row='Laplace Noise Scale', kind='violin',
    #                 inner='quartile', bw=.5, height=6, aspect=max((num_alphas - 2) * 0.3, 1), split=True, legend='full', data=df)
    g = sns.relplot(
        x='Total Number of Samples',
        y=y,
        hue='Number of Real Samples',
        row='Minimised\nDivergence',
        col='Laplace Noise Scale',
        kind='line',
        height=6,
        aspect=max((num_alphas - 2) * 0.3, 1),
        data=df,
        legend='full',
        palette=sns.color_palette("Set1", len(df["Number of Real Samples"].unique()))
    )
    g.fig.suptitle(y + ' comparison', y=1.05)
    g.set(yscale='log')
    g.savefig(plot_dir + '/' + y + '.png')
    plt.close()


def plot_metric(df, metric, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''
    df2 = df.loc[df['Total Number of Samples'] == df['Number of Real Samples']]
    df2.loc[:, 'Number of Real Samples'] = 'Varying'
    df = df.append(df2)
    df = df.melt(['Number of Real Samples', 'Total Number of Samples', 'Laplace Noise Scale'], var_name='Minimised\nDivergence', value_name=metric)
    plot(df, metric, plot_dir, len(df['Number of Real Samples'].unique()) * 2)


def append_data_frames(directory):
    '''
    Scans outputs directory for all iterations matching the passed path and appends all of the contained dataframes that have matching priors
    '''
    temp = None
    for fp in os.listdir(directory):
        try:
            with open(f'{directory}/{fp}', 'rb') as f:
                if temp is None:
                    temp = pickle.load(f)
                else:
                    temp = temp.append(pickle.load(f))
        except Exception:
            pass
    return temp.reset_index(drop=True)


def plot_all_k(exp):
    '''
    Helper function to plot all metrics for a full experiment's iterations
    '''
    df = append_data_frames(f"outputs/{exp}")
    df = df.loc[df['Total Number of Samples'] != 0]
    plot_metric(df.filter(regex='Log Loss|Number of|Laplace Noise'), 'Log Loss', f"plots/{exp}")
    plot_metric(df.filter(regex='KLD|Number of|Laplace Noise'), 'KLD', f"plots/{exp}")
    plot_metric(df.filter(regex='HellingerD|Number of|Laplace Noise'), 'HellingerD', f"plots/{exp}")
    plot_metric(df.filter(regex='TVD|Number of|Laplace Noise'), 'TVD', f"plots/{exp}")
    plot_metric(df.filter(regex='WassersteinD|Number of|Laplace Noise'), 'WassersteinD', f"plots/{exp}")


if __name__ == "__main__":
    plot_all_k("sebexp_05-09-2020_19.58.30")
