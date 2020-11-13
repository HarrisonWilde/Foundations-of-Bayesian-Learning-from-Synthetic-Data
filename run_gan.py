import numpy as np
from python_src.gan.PATE_GAN import PATE_GAN
import argparse
import pandas as pd


# gpu_frac = 0.5
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
# set_session(tf.Session(config=config))

def from_dummies(
    data,
    prefix = None,
    prefix_sep = "_",
    dtype = "category",
):
    """
    The inverse transformation of ``pandas.get_dummies``.
    .. versionadded:: 1.1.0
    Parameters
    ----------
    data : DataFrame
        Data which contains dummy indicators.
    prefix : list-like, default None
        How to name the decoded groups of columns. If there are columns
        containing `prefix_sep`, then the part of their name preceding
        `prefix_sep` will be used (see examples below).
    prefix_sep : str, default '_'
        Separator between original column name and dummy variable.
    dtype : dtype, default 'category'
        Data dtype for new columns - only a single data type is allowed.
    Returns
    -------
    DataFrame
        Decoded data.
    See Also
    --------
    get_dummies : The inverse operation.
    Examples
    --------
    Say we have a dataframe where some variables have been dummified:
    >>> df = pd.DataFrame(
    ...     {
    ...         "baboon": [0, 0, 1],
    ...         "lemur": [0, 1, 0],
    ...         "zebra": [1, 0, 0],
    ...     }
    ... )
    >>> df
       baboon  lemur  zebra
    0       0      0      1
    1       0      1      0
    2       1      0      0
    We can recover the original dataframe using `from_dummies`:
    >>> pd.from_dummies(df, prefix='animal')
      animal
    0  zebra
    1  lemur
    2 baboon
    If our dataframe already has columns with `prefix_sep` in them,
    we don't need to pass in the `prefix` argument:
    >>> df = pd.DataFrame(
    ...     {
    ...         "animal_baboon": [0, 0, 1],
    ...         "animal_lemur": [0, 1, 0],
    ...         "animal_zebra": [1, 0, 0],
    ...         "other": ['a', 'b', 'c'],
    ...     }
    ... )
    >>> df
       animal_baboon  animal_lemur  animal_zebra other
    0              0             0             1     a
    1              0             1             0     b
    2              1             0             0     c
    >>> pd.from_dummies(df)
      other  animal
    0     a   zebra
    1     b   lemur
    2     c  baboon
    """
    if dtype is None:
        dtype = "category"

    columns_to_decode = [i for i in data.columns if prefix_sep in i]
    if not columns_to_decode:
        if prefix is None:
            raise ValueError(
                "If no columns contain `prefix_sep`, you must "
                "pass a value to `prefix` with which to name "
                "the decoded columns."
            )
        # If no column contains `prefix_sep`, we prepend `prefix` and
        # `prefix_sep` to each column.
        out = data.rename(columns=lambda x: f"{prefix}{prefix_sep}{x}").copy()
        columns_to_decode = out.columns
    else:
        out = data.copy()

    data_to_decode = out[columns_to_decode]

    if prefix is None:
        # If no prefix has been passed, extract it from columns containing
        # `prefix_sep`
        seen: Set[str] = set()
        prefix = []
        for i in columns_to_decode:
            i = i.split(prefix_sep)[0]
            if i in seen:
                continue
            seen.add(i)
            prefix.append(i)
    elif isinstance(prefix, str):
        prefix = [prefix]

    # Check each row sums to 1 or 0
    def _validate_values(data):
        if (data.sum(axis=1) != 1).any():
            raise ValueError(
                "Data cannot be decoded! Each row must contain only 0s and "
                "1s, and each row may have at most one 1."
            )

    for prefix_ in prefix:
        cols, labels = (
            [
                i.replace(x, "")
                for i in data_to_decode.columns
                if prefix_ + prefix_sep in i
            ]
            for x in ["", prefix_ + prefix_sep]
        )
        if not cols:
            continue
        _validate_values(data_to_decode[cols])
        out = out.drop(cols, axis=1)
        out[prefix_] = pd.Series(
            np.array(labels)[np.argmax(data_to_decode[cols].to_numpy(), axis=1)],
            dtype=dtype,
        )
    return out


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
    parser.add_argument("--no_split", "-ns", action="store_true", help="Disable test train split mode")
    parser.add_argument("--iter", type=int, default=20000)
    parser.add_argument("--epsilon", type=float, default=6.)
    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--teachers", type=int, default=50)
    parser.add_argument("--targets", nargs='+', help="Name of response var when using csv as input.")
    parser.add_argument("--separator", default=',', help="Separator for the input csv file.")
    return parser.parse_args()


def run(path, name, targets, sep, num_teachers, niter, epsilon, delta, split, no_split, b):

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
        df.loc[df.y.isin([2, 3, 4, 5]), "y"] = 0
        df.loc[df.y.isin([1]), "y"] = 1
        df = df.drop(columns=["Unnamed: 0"], axis=1)
    elif name == "kag_cervical_cancer":
        df = df.fillna(df.mean())
    elif name == "framingham":
        df = df.fillna(df.mean())
    elif name == "kag_creditcard":
        df = df.drop(columns=["Time"], axis=1)
    elif name == "gcse":
        students = df["student"]
        df.drop("student", axis=1, inplace=True)
        df = pd.get_dummies(df, columns=["school"])

    features = list(df.columns)
    for lbl in targets:
        assert lbl in features
        features.remove(lbl)

    if no_split:
        out_name = f"{name}_{''.join(targets)}_eps{str(epsilon)}"

        x_new, y_new = PATE_GAN(
            df[features].values,
            df[targets].values,
            None,
            None,
            epsilon,
            delta,
            niter,
            num_teachers,
            no_split
        )

        cols = features
        cols.extend(targets)

        df_new = pd.DataFrame(
            np.hstack(
                [x_new,
                 y_new.reshape(len(y_new), -1)]),
            columns=cols)

        if name == "gcse":
            df["school"] = df.filter(regex=(r"school_")).idxmax(axis=1).str.replace("school_", "")
            df.drop(list(df.filter(regex='school_')), axis=1, inplace=True)
            df["student"] = students
            df_new["school"] = df_new.filter(regex=(r"school_")).idxmax(axis=1).str.replace("school_", "")
            df_new.drop(list(df_new.filter(regex='school_')), axis=1, inplace=True)
            df_new["student"] = df_new.groupby(["school"]).cumcount() + 1

        df.to_csv(f'{path}splits/{out_name}_real.csv', index=False)
        df_new.to_csv(f'{path}splits/{out_name}_synth.csv', index=False)

        if b is not None:
            b.set_description_str('PATE-GAN training and generation complete.')

        return df, df_new

    else:
        out_name = f"{name}_{''.join(targets)}_split{str(split)}_eps{str(epsilon)}"

        df_train, df_test = tt_split(df, split)

        x_train_new, y_train_new, x_test_new, y_test_new = PATE_GAN(
            df_train[features].values,
            df_train[targets].values,
            df_test[features].values,
            df_test[targets].values,
            epsilon,
            delta,
            niter,
            num_teachers,
            no_split
        )

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

        df_train.to_csv(f'{path}splits/{out_name}_real_train.csv', index=False)
        df_test.to_csv(f'{path}splits/{out_name}_real_test.csv', index=False)
        df_train_new.to_csv(f'{path}splits/{out_name}_synth_train.csv', index=False)
        df_test_new.to_csv(f'{path}splits/{out_name}_synth_test.csv', index=False)

        if b is not None:
            b.set_description_str('PATE-GAN training and generation complete.')

        return df_train, df_test, df_train_new, df_test_new


if __name__ == '__main__':

    args = init_arg()
    run(
        args.path_to_data, args.dataset_name, args.targets, args.separator,
        args.teachers, args.iter, args.epsilon, args.delta, args.split, args.no_split, None
    )