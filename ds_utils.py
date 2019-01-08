import itertools
import operator
from functools import reduce
from typing import Iterable

import numpy as np
import pandas as pd


def min_max_scaler(x):
    return (x - x.min()) / (x.max() - x.min())


def weighted_average(data, weights=None, axis=1):
    """Calculates the weighted average(s) of an array.

    Args:
        row: Array of data to calculate averages for.
        weights: Array of weights to use for averaging. If None (default), equal weighting will be used.
        axis: Axis along which to average. If None, averaging is done over the flattened array. Default is 1.

    """
    cleaned_data = np.ma.masked_array(data, np.isnan(data))
    weighted_average = np.average(cleaned_data, weights=weights, axis=1)

    return weighted_average.filled(np.nan)


def binning(col, bins, labels=None, bin_type="quantile", include_lowest=True, duplicates='drop', **kwargs):
    """
    """
    # Here because pd.cut/pd.qcut bin/q errors aren't clear.
    if isinstance(bins, int):
        if bins != len(labels):
            raise ValueError(f"The number of labels ({len(labels)} provided) must match the number of bins ({bins} provided).")
    elif isinstance(bins, (tuple, list)):
        if len(bins) != len(labels) + 1:
            raise ValueError(f"The number of labels ({len(labels)} provided) must be one fewer than the number of bins ({len(bins)} provided).")

    if bin_type == "quantile":
        return pd.qcut(col.rank(method='first'), q=bins, labels=labels, duplicates=duplicates, **kwargs)
    elif bin_type == "absolute":
        return pd.cut(col.rank(method='first'), bins=bins, labels=labels, include_lowest=include_lowest, duplicates=duplicates, **kwargs)
    else:
        raise KeyError(f"The type '{bin_type}' is an invalid option. "
                       f"Please choose from: {', '.join({'quantile', 'absolute'})}")


def check_outsized_corrs(df: pd.DataFrame, thresh: np.number = 0.80) -> pd.Series:
    """Identify outsized correlations among columns in `df`.

    Args:
        df: Data source from which to look for high pairwise column correlations.
        thresh: absolute threshold for determining which correlations are "outsized".

    Returns:
        A multiindex pd.Series with pairs of column names that have outsized correlations.

    """
    corr_matrix = df.corr()
    corr_ex_self = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Create long-form df of outsized corrs
    long_form_corrs = corr_ex_self.unstack()

    # Only return values outside the threshold
    outsized_corrs = long_form_corrs[~long_form_corrs.between(-thresh, thresh)].dropna()

    return outsized_corrs.sort_values(kind="quicksort")


def generate_combinations(items: list, lengths: Iterable, max_len: int=20):
    """Generate all combinations of `items` of all of the specified lengths.

    Args:
        items: Items to be used in creating combinations.
               If output is meant to be used with generate_interacted_features(),
               each item should reflect a column name in a pd.DataFrame.
        lengths:

    Returns:
        List of tuples reflecting combinations of `items` of selected `lengths`.

    Example:
        Basic usage:
        >>> generate_combinations(items=["a", "b", "c", "d"], lengths=[2, 3, 4, 5])

        Often it will be easier to use a range to specify lengths:
        >>> generate_combinations(items=["a", "b", "c", "d"], lengths=range(2, 6))

    See also:
        `generate_av_interaction_pairs`
        `generate_interacted_features`

    """
    if max(lengths) > max_len:
        raise ValueError(f"{max(lengths)} is longer than the longest allowed combination "
                         f"of items set by `max_len`. `max_len` is important because "
                         f"combination generation is exponential. "
                         f"To avoid this exception, either revise `lengths` (recommended) or raise `max_len`.")
    return [i for i in itertools.chain.from_iterable(itertools.combinations(items, length) for length in lengths)]


def generate_av_interaction_pairs(avs_to_interact, dummy_delimiter="|"):
    """Create interacted features for model from given attribute value pairs."""
    all_interaction_pairs = generate_combinations(avs_to_interact, range(2, 3))

    # Remove pairs that share the same attribute
    interaction_pairs = []
    for i, pair in enumerate(all_interaction_pairs):
        if pair[0].split(dummy_delimiter)[0] != pair[1].split(dummy_delimiter)[0]:
            interaction_pairs.append(pair)

    return interaction_pairs


def generate_interacted_features(df, interaction_groups, func=operator.mul, dummy_delimiter="|", skip_bad_groups=False):
    """Create interacted features for each group in `interaction_groups`,
       where group elements are names of columns in `df`.

    Args:
        df (pd.DataFrame): A df containing columns from which to create interactions.
        interaction_groups (list of tuples): Each inner tuple is filled with column names to interact.
        func (func): Function used to interact the columns to create the new interacted features.
                     Default is to multiple the columns.
                     To use a custom function, the function must take two pd.Series as input.
        dummy_delimiter (str): str to be used to separate names combined to create the interacted column name.
        skip_bad_groups (bool): If there are columns in `interaction_groups`, determines if this function skips
                                the bad interaction groups or errors.

    Returns:
        A pd.DataFrame with only the new interaction features.

    Example:

    >>> df = pd.DataFrame({"a": [2, 3, 4, 5], "b": [6, 7, 8, 9], "c": [10, 11, 12, 13]})
    >>> generate_interacted_features(df=df,
                                     interaction_groups=[("a", "b"), ("a", "b", "c")],
                                     dummy_delimiter=" * ")
    Out:        a * b  a * b * c
            0     12        120
            1     21        231
            2     32        384
            3     45        585

    """
    bad_columns = set([col for g in interaction_groups for col in g]).difference(df)
    if bad_columns and not skip_bad_groups:
        raise KeyError(f"The following column names from intraction groups are not in the df: "
                       f"{', '.join(bad_columns)}. Please confirm that all columns in these groups exist in `df`. "
                       f"To ignore interactions with bad columns, set the skip_bad_groups flag to True.")

    interactions = {}
    for g in interaction_groups:
        if set(g).difference(df):
            continue
        # Below method is significantly faster than
        # df.loc[:, g].apply(lambda x: reduce(func, x), axis='columns')
        interactions.update({f"{f'{dummy_delimiter}'.join(g)}": reduce(func, [df[col] for col in g])})

    return pd.DataFrame(interactions)
