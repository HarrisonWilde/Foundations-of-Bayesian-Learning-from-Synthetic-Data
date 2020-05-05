import numpy as np
from python_src.gan.PATE_GAN import PATE_GAN
import argparse
import pandas as pd


# gpu_frac = 0.5
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
# set_session(tf.Session(config=config))


def tt_split(df, ratio=0.7):

    idx = np.random.permutation(len(df))
    train_idx = idx[:int(ratio * len(df))]
    test_idx = idx[int(ratio * len(df)):]
    df_train = df.iloc[train_idx, :]
    df_test = df.iloc[test_idx, :]
    return df_train, df_test


def init_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_data", default="data/", help="Path to data folder.")
    parser.add_argument("-i", "--dataset_name", default="creditcard", help="Name of dataset to load (csv file).")
    parser.add_argument("-tts", "--split", type=float, default=0.5)
    parser.add_argument("--iter", type=int, default=20000)
    parser.add_argument("--epsilon", type=float, default=6.)
    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--teachers", type=int, default=50)
    parser.add_argument("--targets", nargs='+', help="Name of response var when using csv as input.")
    parser.add_argument("--separator", default=',', help="Separator for the input csv file.")
    return parser.parse_args()


def run(path, name, targets, sep, num_teachers, niter, epsilon, delta, split, b):

    out_name = name + '_' + ''.join(targets) + '_split' + str(split) + '_ganpate'

    if b is not None:
        b.set_description_str(f'Training PATE-GAN with {num_teachers} teachers, {niter} iterations, delta = {delta}...')

    assert path is not None
    assert targets is not None

    # Dataset specific pre-processing
    if name == "kag_cervical_cancer":
        df = pd.read_csv(f"{path}raw/{name}.csv", na_values='?')
    else:
        df = pd.read_csv(f"{path}raw/{name}.csv")

    if name == "uci_seizures":
        df.y[df.y.isin([2, 3, 4, 5])] = 0
        df.y[df.y.isin([1])] = 1
        df = df.drop(columns=["Unnamed: 0"], axis=1)
    if name == "kag_cervical_cancer":
        df = df.fillna(df.mean())
    if name == "kag_creditcard":
        df = df.drop(columns=["Time"], axis=1)

    features = list(df.columns)
    for lbl in targets:
        assert lbl in features
        features.remove(lbl)

    df_train, df_test = tt_split(df, split)

    x_train_new, y_train_new, x_test_new, y_test_new = PATE_GAN(
        df_train[features].values,
        df_train[targets].values,
        df_test[features].values,
        df_test[targets].values,
        epsilon,
        delta,
        niter,
        num_teachers)

    cols = features
    cols.extend(targets)

    df_train_new = pd.DataFrame(
        np.hstack(
            [x_train_new,
             y_train_new.reshape(len(y_train_new), -1)]),
        columns=cols)

    df_test_new = pd.DataFrame(
        np.hstack(
            [x_test_new,
             y_test_new.reshape(len(y_test_new), -1)]),
        columns=cols)

    df_train.to_csv(f'{path}splits/{out_name}_eps{str(epsilon)}_real_train.csv', index=False)
    df_test.to_csv(f'{path}splits/{out_name}_eps{str(epsilon)}_real_test.csv', index=False)
    df_train_new.to_csv(f'{path}splits/{out_name}_eps{str(epsilon)}_synth_train.csv', index=False)
    df_test_new.to_csv(f'{path}splits/{out_name}_eps{str(epsilon)}_synth_test.csv', index=False)

    if b is not None:
        b.set_description_str('PATE-GAN training and generation complete.')

    return df_train, df_test, df_train_new, df_test_new


if __name__ == '__main__':

    args = init_arg()
    _, _, _, _, = run(args.path_to_data, args.dataset_name, args.targets,
                      args.separator, args.teachers, args.iter, args.epsilon, args.delta, args.split, None)
