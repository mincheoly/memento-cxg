#!/usr/bin/env python
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import tiledb

from memento.cell_census_summary_cube import CUBE_TILEDB_ATTRS_OBS

# Running actual hypothesis test with pre-computed standard errors

DE_TREATMENT = 'cell_type'
DE_COVARIATES = ['dataset_id', 'donor_id', 'assay']
DE_VARIABLES = [DE_TREATMENT] + DE_COVARIATES


def run(cube_path_: str, filter_1: str, filter_2: str) -> Tuple[pd.DataFrame]:
    # Read estimators
    with tiledb.open(cube_path_, 'r') as estimators:
        ct1_df = estimators.query(cond=filter_1, attrs=CUBE_TILEDB_ATTRS_OBS + ['mean', 'sem', 'var', 'selv', 'n_obs']).df[:]
        ct2_df = estimators.query(cond=filter_2, attrs=CUBE_TILEDB_ATTRS_OBS + ['mean', 'sem', 'var', 'selv', 'n_obs']).df[:]
    # convert Pandas columns to `category`, as memory optimization
    for df in [ct1_df, ct2_df]:
        for name, dtype in zip(df.columns, df.dtypes):
            if dtype == 'object':
                df[name] = df[name].astype('category')

    # the method from git@github.com:yelabucsf/scrna-parameter-estimation.git
    # result_1 = compute_cxg_pvalues(cell_type_1_df=ct1_df, cell_type_2_df=ct2_df)
    result_1 = None

    # the method from hypothesis_test.ipynb
    # TODO: This method only takes 1 set of cells!?
    estimators = pd.concat([ct1_df, ct2_df])
    cell_counts, design, features, mean, se_mean = setup(estimators)
    result_2 = compute_hypothesis_test(cell_counts, design, features[:100], mean, se_mean)

    return result_1, result_2


#
# The below code section is adapted from
# https://github.com/yelabucsf/scrna-parameter-estimation/blob/master/lupus/cxg_comparison/cellxgene_comparison.ipynb
#

def compute_cxg_pvalues(cell_type_1_df: pd.DataFrame, cell_type_2_df: pd.DataFrame) -> pd.DataFrame:
    def _fit_mv_regressor(mean, var):
        """
            Perform regression of the variance against the mean.
        """

        cond = (mean > 0) & (var > 0)
        m, v = np.log(mean[cond]), np.log(var[cond])

        poly = np.polyfit(m, v, 2)
        return poly
        #f = np.poly1d(z)

    def _residual_variance(mean, var, mv_fit):
        cond = (mean > 0) & (var > 0)
        rv = np.zeros(mean.shape) * np.nan

        f = np.poly1d(mv_fit)
        with np.errstate(invalid='ignore'):
            rv[cond] = np.exp(np.log(var[cond]) - f(np.log(mean[cond])))
        return rv

    def compute_residual_variance(df):
        m = df['mean']
        v = df['var']
        mv_fit = _fit_mv_regressor(m, v)
        rv = _residual_variance(m, v, mv_fit)
        df['res_var'] = rv

    compute_residual_variance(cell_type_1_df)
    compute_residual_variance(cell_type_2_df)

    # TODO: This merge operation only makes sense if each feature_id value is associated with a distinct tuple of
    #  treatment/covariate values unique on each df. How to combine?
    assert (cell_type_1_df.value_counts(subset='feature_id') == 1).all
    # assert cell_type_2_df.value_counts(subset='feature_id') == 1
    merged = cell_type_1_df.merge(cell_type_2_df, on='feature_id', suffixes=('_ct1', '_ct2'), copy=False)

    lfc = np.log(merged['mean_ct2'].values / merged['mean_ct1'].values)
    log_mean_se_1 = (np.log(merged['mean_ct1'] + merged['sem_ct1']) - np.log(
        merged['mean_ct1'] - merged['sem_ct1'])) / 2
    log_mean_se_2 = (np.log(merged['mean_ct2'] + merged['sem_ct2']) - np.log(
        merged['mean_ct2'] - merged['sem_ct2'])) / 2
    se_lfc = np.sqrt((log_mean_se_2 ** 2 + log_mean_se_1 ** 2)).values
    de_pvalues = stats.norm.sf(np.abs(lfc), loc=0, scale=se_lfc) * 2

    dv_lfc = np.log(merged['res_var_ct2'].values / merged['res_var_ct1'].values)
    se_dv_lfc = np.sqrt((merged['selv_ct1'] ** 2 + merged['selv_ct2'] ** 2)).values
    dv_pvalues = stats.norm.sf(np.abs(dv_lfc), loc=0, scale=se_dv_lfc) * 2

    cxg_results = pd.DataFrame(
        zip(
            merged['feature_id'].values,
            lfc,
            se_lfc,
            de_pvalues,
            dv_lfc,
            se_dv_lfc,
            dv_pvalues),
        columns=['gene', 'cxg_de_coef', 'cxg_de_se', 'cxg_de_pval', 'cxg_dv_coef', 'cxg_dv_se', 'cxg_dv_pval'])
    return cxg_results

#
# This code section is adapted from the notebook hypothesis_test.ipynb:
#


def compute_hypothesis_test(cell_counts, design, features, mean, se_mean) -> pd.DataFrame:
    de_result = []
    # TODO: parallelize, as needed
    for feature in features:
        m = mean[feature].values
        sem = se_mean[feature].values

        # Transform to log space (alternatively can resample in log space)
        lm = np.log(m)
        selm = (np.log(m + sem) - np.log(m - sem)) / 2

        coef, z, pv = de_wls(design.values, lm, cell_counts, selm ** 2)
        de_result.append((feature, coef, z, pv))

    return pd.DataFrame(de_result, columns=['feature_id', 'coef', 'z', 'pval']).set_index('feature_id')


# TODO: Validate this refactored code is equivalent; it uses Pandas multi-level indexes
# def setup(estimators):
#     features = estimators['feature_id'].drop_duplicates().tolist()
#
#     # To create a pivot table, it is necessary to further aggregate the cube on the logical dimensions that are not
#     # DE_VARIABLES
#     # TODO: Is it valid to aggregate `mean` and `sem` using `mean` func?
#     mean = estimators.pivot_table(index=DE_VARIABLES, columns='feature_id', values='mean').fillna(1e-3)
#     se_mean = estimators.pivot_table(index=DE_VARIABLES, columns='feature_id', values='sem').fillna(1e-4)
#
#     groups = estimators.drop_duplicates(DE_VARIABLES).set_index(DE_VARIABLES)
#     # the non-dupped `n_obs` values in `groups` are okay to use, since it is the same value for every cell of a group
#     cell_counts = groups['n_obs'].sort_index().values
#     design = pd.get_dummies(groups, drop_first=True).astype(int)
#
#     assert groups.shape[0] == mean.shape[0]
#
#     return cell_counts, design, features, mean, se_mean

def setup(estimators):
    names = estimators[DE_TREATMENT].copy()
    for col in DE_COVARIATES:
        names += '_' + estimators[col]
    estimators['group_name'] = names.tolist()
    features = estimators['feature_id'].drop_duplicates().tolist()
    groups = estimators.drop_duplicates(subset='group_name').set_index('group_name')
    design = pd.get_dummies(groups[DE_VARIABLES], drop_first=True).astype(int)
    # TODO: Is it valid to aggregate `mean` and `sem` using `mean` func?
    mean = estimators.pivot_table(index='group_name', columns='feature_id', values='mean').fillna(1e-3)
    se_mean = estimators.pivot_table(index='group_name', columns='feature_id', values='sem').fillna(1e-4)
    cell_counts = groups['n_obs'].sort_index().values

    pd.options.display.max_columns = None
    print(estimators)
    print(groups)
    print(f"estimators len={len(estimators)}, groups len={len(groups)}, cell_counts len={len(cell_counts)}")

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

    # we have all the other coeffs for the other covariates here as well
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

    if len(sys.argv) < 5:
        print('Usage: python hypothesis_test.py <filter_1> <filter_2> <cube_path> <csv_output_path>')
        sys.exit(1)

    filter_1_arg, filter_2_arg, cube_path_arg, csv_output_path_arg = sys.argv[1:5]

    de_result_1, de_result_2 = run(cube_path_arg, filter_1_arg, filter_2_arg)

    # Output DE result
    print(de_result_1, de_result_2)
    #de_result.to_csv(csv_output_path_arg)

