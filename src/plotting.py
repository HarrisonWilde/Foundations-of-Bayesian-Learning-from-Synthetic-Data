import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sns.set(rc={'figure.figsize': (10, 10)})


def explode(df, lst_cols, fill_value='', preserve_index=False):
    '''
    Used to deal with df columns that contain lists length n, explodes them into n rows each
    '''

    if (lst_cols is not None and len(lst_cols) > 0 and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]

    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    # preserve original index values
    idx = np.repeat(df.index.values, lens)

    # create "exploded" DF
    res = (pd.DataFrame({
            col: np.repeat(df[col].values, lens) for col in idx_cols}, index=idx
        ).assign(**{col: np.concatenate(df.loc[lens > 0, col].values) for col in lst_cols}))

    # revert the original index order
    res = res.sort_index()

    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)

    return res


def plot_alpha(df, y, title, plot_dir, num_alphas, output_timestamp):
    '''
    Generic plot function for plotting alpha on x-axis and metric on y-axis
    '''

    # g = sns.catplot(x='Alpha', y=y, hue='Minimised\nDivergence', row='Laplace Noise Scale', kind='violin',
    #                 inner='quartile', bw=.5, height=6, aspect=max((num_alphas - 2) * 0.3, 1), split=True, legend='full', data=df)
    g = sns.relplot(x='Alpha', y=y, hue='Minimised\nDivergence', row='Laplace Noise Scale', kind='line',
                    height=6, aspect=max((num_alphas - 2) * 0.3, 1), legend='full', data=df)
    g.fig.suptitle(y + ' comparison for ' + title, y=1.05)
    g.set(yscale='log')
    g.savefig(plot_dir + '/' + y + output_timestamp + ' ' + title + '.png')
    plt.close()


def plot_k(df, y, title, plot_dir, num_alphas, output_timestamp):
    '''
    Generic plot function for plotting number of samples / contaminated samples on x-axis and metric on y-axis
    '''
    # g = sns.catplot(x='Alpha', y=y, hue='Minimised\nDivergence', row='Laplace Noise Scale', kind='violin',
    #                 inner='quartile', bw=.5, height=6, aspect=max((num_alphas - 2) * 0.3, 1), split=True, legend='full', data=df)
    g = sns.relplot(x='Total Number of Samples', y=y, hue='Number of Real Samples', row='Minimised\nDivergence',
                    col='Laplace Noise Scale', kind='line', height=6, aspect=max((num_alphas - 2) * 0.3, 1), legend='full', data=df)
    g.fig.suptitle(y + ' comparison for ' + title, y=1.05)
    g.set(yscale='log')
    g.savefig(plot_dir + '/' + y + output_timestamp + ' ' + title + '.png')
    plt.close()


def plot_pdfs(df, output_timestamp, plot_dir):
    '''
    Plots comparison curves on x-axis of y-tilde and y-axis of value of posterior predictive
    '''

    df = df.melt('Y Tilde Value', var_name='PDF', value_name='p(y tilde | y)')
    df = explode(df, ['Y Tilde Value', 'p(y tilde | y)'])
    g = sns.lineplot(x='Y Tilde Value', y='p(y tilde | y)', hue='PDF', data=df)
    g.get_figure().savefig(plot_dir + '/pdfline' + output_timestamp + '.png')
    plt.close()


def plot_metric(df, metric, num_alphas, prior, dgp, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''

    title = f'''y[{dgp[2]}] ~ Norm({dgp[0]}, {dgp[1]}), beta = {prior[3]}, beta w = {prior[6]}, w = {prior[5]}
    Prior: sigma2 ~ InvGamma({prior[1]}, {prior[2]}), mu ~ Norm({prior[0]}, {prior[4]} * sigma2)'''
    df = df.melt(['Alpha', 'Laplace Noise Scale'], var_name='Minimised\nDivergence', value_name=metric)
    plot_alpha(df, metric, title, plot_dir, num_alphas, output_timestamp)


def plot_metric_(df, metric, title, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''

    df = df.melt(['Alpha', 'Laplace Noise Scale'], var_name='Minimised\nDivergence', value_name=metric)
    plot_alpha(df, metric, title, plot_dir, len(df.Alpha.unique()), output_timestamp)


def plot_metric_k(df, metric, num_alphas, prior, dgp, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''

    title = f'''y[{dgp[2]}] ~ Norm({dgp[0]}, {dgp[1]}), beta = {prior[3]}, beta w = {prior[6]}, w = {prior[5]}
    Prior: sigma2 ~ InvGamma({prior[1]}, {prior[2]}), mu ~ Norm({prior[0]}, {prior[4]} * sigma2)'''
    df2 = df.loc[df['Total Number of Samples'] == df['Number of Real Samples']]
    df2['Number of Real Samples'] = 'Varying'
    df = df.append(df2)
    df = df.melt(['Number of Real Samples', 'Total Number of Samples', 'Laplace Noise Scale'], var_name='Minimised\nDivergence', value_name=metric)
    plot_k(df, metric, title, plot_dir, num_alphas, output_timestamp)


def plot_metric_k_(df, metric, title, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''

    df2 = df.loc[df['Total Number of Samples'] == df['Number of Real Samples']]
    df2['Number of Real Samples'] = 'Varying'
    df = df.append(df2)
    df = df.melt(['Number of Real Samples', 'Total Number of Samples', 'Laplace Noise Scale'], var_name='Minimised\nDivergence', value_name=metric)
    plot_k(df, metric, title, plot_dir, len(df['Number of Contaminated Samples'].unique()), output_timestamp)
