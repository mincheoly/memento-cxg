#!/usr/bin/env python
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import tiledb

from memento.cell_census_summary_cube import CUBE_TILEDB_ATTRS_OBS, CUBE_LOGICAL_DIMS_OBS

# Running actual hypothesis test with pre-computed standard errors

DE_TREATMENT = 'cell_type'
DE_COVARIATES = ['dataset_id', 'donor_id', 'assay']


def run(cube_path_: str, filter_: str) -> pd.DataFrame:
    estimators_df = query_estimators(cube_path_, filter_)
    cell_counts, design, features, mean, se_mean = setup(estimators_df, DE_TREATMENT)
    return compute_hypothesis_test(cell_counts, design, features[:100], mean, se_mean)


def query_estimators(cube_path_, filter_) -> pd.DataFrame:
    with tiledb.open(cube_path_, 'r') as estimators:
        estimators_df = estimators.query(cond=filter_,
                                         attrs=CUBE_TILEDB_ATTRS_OBS + ['mean', 'sem', 'var', 'selv', 'n_obs']).df[:]

        # TODO: convert Pandas columns to `category`, as memory optimization (shouldn't be needed if upgrade census/tiledb-soma versions)
        # for name, dtype in zip(estimators_df.columns, estimators_df.dtypes):
        #     if dtype == 'object':
        #         estimators_df[name] = estimators_df[name].astype('category')
        return estimators_df


def compute_hypothesis_test(cell_counts, design, features, mean, se_mean) -> pd.DataFrame:
    de_result = []
    # TODO: parallelize, as needed
    for feature in features:
        m = mean[feature].values
        sem = se_mean[feature].values

        # Transform to log space (alternatively can resample in log space)
        lm = np.log(m)
        selm = (np.log(m + sem) - np.log(m - sem)) / 2

        coef, z, pv = de_wls(X=design.values, y=lm, n=cell_counts, v=selm ** 2)
        de_result.append((feature, coef, z, pv))

    return pd.DataFrame(de_result, columns=['feature_id', 'coef', 'z', 'pval']).set_index('feature_id')


def setup(estimators, treatment_variable: str) -> Tuple[np.array, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distinct_treatment_values = estimators[[treatment_variable]].nunique()[0]
    assert distinct_treatment_values == 2, "treatment must have exactly 2 distinct values"

    # make treatment variable be in the first column of the design matrix
    variables = ([treatment_variable] +
                 [covariate for covariate in CUBE_LOGICAL_DIMS_OBS if covariate != treatment_variable])

    mean = estimators.pivot_table(index=variables, columns='feature_id', values='mean').fillna(1e-3)
    se_mean = estimators.pivot_table(index=variables, columns='feature_id', values='sem').fillna(1e-4)

    groups = estimators[variables + ['n_obs']].drop_duplicates(variables)
    cell_counts = groups['n_obs'].values
    design = pd.get_dummies(groups[variables], drop_first=True, dtype=int)

    features = estimators['feature_id'].drop_duplicates().tolist()

    assert len(cell_counts) == len(groups)
    assert design.shape[0] == len(groups)
    assert mean.shape == (len(groups), len(features))
    assert se_mean.shape == (len(groups), len(features))

    return cell_counts, design, features, mean, se_mean


# assume variable of interest is the first column of X; should parameterize
def de_wls(X, y, n, v):
    """
    Perform DE for each gene using Weighted Least Squares (i.e., a weighted Linear Regression model)
    """

    from sklearn.linear_model import LinearRegression

    # fit WLS using sample_weights
    WLS = LinearRegression()
    WLS.fit(X, y, sample_weight=n)

    # note: we have all the other coeffs for the other covariates here as well
    treatment_col = 0
    coef = WLS.coef_[treatment_col]

    W = np.diag(1 / v)
    beta_var_hat = np.diag(np.linalg.pinv(X.T @ W @ X))
    se = np.sqrt(beta_var_hat[treatment_col])

    z = coef / se
    pv = stats.norm.sf(np.abs(z)) * 2

    return coef, z, pv


# Script entrypoint
if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Usage: python diff_expr.py <filter> <cube_path> <csv_output_path>')
        sys.exit(1)

    filter_arg, cube_path_arg, csv_output_path_arg = sys.argv[1:4]

    de_result = run(cube_path_arg, filter_arg)

    # Output DE result
    print(de_result)
    de_result.to_csv(csv_output_path_arg)

