import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize': (10, 10)})


def plot(df, y, title, plot_dir, num_alphas, output_timestamp):
    '''
    Generic plot function for plotting number of samples / contaminated samples on x-axis and metric on y-axis
    '''
    # g = sns.catplot(x='Alpha', y=y, hue='Minimised\nDivergence', row='Laplace Noise Scale', kind='violin',
    #                 inner='quartile', bw=.5, height=6, aspect=max((num_alphas - 2) * 0.3, 1), split=True, legend='full', data=df)
    g = sns.relplot(x='Total Number of Samples', y=y, hue='Number of Real Samples', row='Minimised\nDivergence',
                    kind='line', height=6, aspect=max((num_alphas - 2) * 0.3, 1), legend='full', data=df)
    g.fig.suptitle(y + ' comparison for ' + title, y=1.05)
    g.set(yscale='log')
    g.savefig(plot_dir + '/' + y + output_timestamp + ' ' + title + '.png')
    plt.close()


def plot_metric(df, metric, prior, output_timestamp, plot_dir):
    '''
    Passed data frame is exploded and transformed to be passed to plot
    '''
    title = ""
    df2 = df.loc[df['Total Number of Samples'] == df['Number of Real Samples']]
    df2.loc[:, 'Number of Real Samples'] = 'Varying'
    df = df.append(df2)
    df = df.melt(['Number of Real Samples', 'Total Number of Samples'], var_name='Metric', value_name=metric)
    plot(df, metric, title, plot_dir, len(df['Number of Real Samples'].unique()) * 2, output_timestamp)
