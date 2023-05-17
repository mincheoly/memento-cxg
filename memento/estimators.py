import logging

import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats


def bin_size_factor(size_factor, num_bins=30):
    """ Bin the size factors to speed up bootstrap. """
    
    binned_stat = stats.binned_statistic(size_factor, size_factor, bins=num_bins, statistic='mean')
    bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
    approx_sf = binned_stat[0][bin_idx-1]
    max_sf = size_factor.max()
    approx_sf[size_factor == max_sf] = max_sf
    
    return approx_sf


def fill_invalid(val, group_name):
    """ Fill invalid entries by randomly selecting a valid entry. """

    # negatives and nan values are invalid values for our purposes
    invalid_mask = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
    num_invalid = invalid_mask.sum()

    if num_invalid == val.shape[0]:
        # if all values are invalid, there are no valid values to choose from, so return all nans
        logging.warning(f"all bootstrap variances are invalid for group {group_name}")
        return np.full(shape=val.shape, fill_value=np.nan)

    val[invalid_mask] = np.random.choice(val[~invalid_mask], num_invalid)
    
    return val


def unique_expr(expr, size_factor):
    """
        Find (approximately) unique combinations of expression values and size factors.
        The random component is for mapping (expr, size_factor) to a single number.
        This can certainly be performed more efficiently using sparsity.
    """

    code = expr.dot(np.random.random(expr.shape[1]))
    approx_sf = size_factor

    code += np.random.random() * approx_sf

    _, index, count = np.unique(code, return_index=True, return_counts=True)

    expr_to_return = expr[index].toarray()

    return 1 / approx_sf[index].reshape(-1, 1), 1 / approx_sf[index].reshape(-1, 1) ** 2, expr_to_return, count


def compute_mean(X: np.array, q: float, sample_mean: float, variance: float, size_factors: np.array):
    """ Inverse variance weighted mean. """
    cell_variance = (1-q) / size_factors * sample_mean + variance

    norm_X = X * (1 / size_factors)

    if cell_variance.sum() == 0:
        logging.warning(f"compute_mean(): weights sum is zero; {sample_mean=}, {variance=}, size_factors count={len(size_factors)}")
        cell_variance = None

    return np.average(np.nan_to_num(norm_X), weights=cell_variance)


def compute_sem(variance, n_obs: int):
    """ Approximate standard error of the mean. """

    if variance < 0:
        # Avoids a numpy warning, returns same result
        return np.nan

    return np.sqrt(variance/n_obs)



def compute_variance(X: sparse.csc_matrix, q: float, size_factor: np.array, group_name=None):
    """ Compute the variances. """

    n_obs = X.shape[0]
    row_weight = (1 / size_factor).reshape([1, -1])
    row_weight_sq = (1 / size_factor ** 2).reshape([1, -1])

    mm_M1 = sparse.csc_matrix.dot(row_weight, X).ravel() / n_obs
    mm_M2 = sparse.csc_matrix.dot(row_weight_sq, X.power(2)).ravel() / n_obs - (1 - q) * sparse.csc_matrix.dot(
        row_weight_sq, X).ravel() / n_obs

    mean = mm_M1
    variance = (mm_M2 - mm_M1 ** 2)

    if variance < 0:
        logging.warning(f"negative variance ({variance}) for group {group_name}: {X.data}")
        variance = mean

    return float(mean), float(variance)


def compute_bootstrap_variance(
        unique_expr: np.array,
        bootstrap_freq: np.array,
        q: float,
        n_obs: int,
        inverse_size_factor: np.array,
        inverse_size_factor_sq: np.array
):
    """ Compute the bootstrapped variances for a single gene expression frequencies."""

    mm_M1 = (unique_expr * bootstrap_freq * inverse_size_factor).sum(axis=0) / n_obs
    mm_M2 = (unique_expr ** 2 * bootstrap_freq * inverse_size_factor_sq - (
                1 - q) * unique_expr * bootstrap_freq * inverse_size_factor_sq).sum(axis=0) / n_obs

    variance = mm_M2 - mm_M1 ** 2
    return variance


def compute_sev(
        X: sparse.csc_matrix,
        q: float,
        approx_size_factor: np.array,
        num_boot: int = 5000,
        group_name: tuple = ()
):
    """ Compute the standard error of the variance. """
    
    if X.max() < 2:
        return np.nan, np.nan
    
    n_obs = X.shape[0]
    inv_sf, inv_sf_sq, expr, counts = unique_expr(X, approx_size_factor)

    gen = np.random.Generator(np.random.PCG64(5))
    gene_rvs = gen.multinomial(n_obs, counts / counts.sum(), size=num_boot).T

    var = compute_bootstrap_variance(
        unique_expr=expr,
        bootstrap_freq=gene_rvs,
        n_obs=n_obs,
        q=q,
        inverse_size_factor=inv_sf,
        inverse_size_factor_sq=inv_sf_sq
    )

    var = fill_invalid(var, group_name)

    sev = np.nanstd(var)
    selv = np.nanstd(np.log(var))

    return sev, selv
