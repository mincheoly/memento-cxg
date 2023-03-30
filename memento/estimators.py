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


def fill_invalid(val):
	""" Fill invalid entries by randomly selecting a valid entry. """
	
	invalid_mask = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
	num_invalid = invalid_mask.sum()
	
	if num_invalid == val.shape[0]:
		# TODO: Returning None causes failure. What does it mean when all values are invalid and how should this be handled?
		return np.zeros(shape=val.shape)
		# return None

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
		
	code += np.random.random()*approx_sf
	
	_, index, count = np.unique(code, return_index=True, return_counts=True)
	expr_to_return = expr[index].toarray()
	
	return (
		1/approx_sf[index].reshape(-1, 1),
		1/approx_sf[index].reshape(-1, 1)**2,
		expr_to_return,
		count
	)


def compute_mean(
	X: sparse.csc_matrix,
	q: float,
	sample_mean,
	variance,
	size_factor: np.array
):
	""" Inverse variance weighted mean. """

	cell_variance = (1-q)/size_factor*sample_mean + variance

	norm_X = X.multiply(1/size_factor.reshape(-1, 1))
	# TODO: OK to us np.ma methods?
	mean = np.ma.average(norm_X.todense().A1, weights=cell_variance)

	return mean


def compute_sem(
	X: sparse.csc_matrix,
	variance
):
	""" Approximate standard error of the mean. """
	
	sem = np.sqrt(variance/X.shape[0])
	
	return sem
	
	
def compute_variance(
	X: sparse.csc_matrix,
	q: float,
	size_factor: np.array,
):
	""" Compute the variances. """
	
	n_obs = X.shape[0]
	row_weight = (1/size_factor).reshape([1, -1])
	row_weight_sq = (1/size_factor**2).reshape([1, -1])
	
	mm_M1 = sparse.csc_matrix.dot(row_weight, X).ravel()/n_obs
	mm_M2 = sparse.csc_matrix.dot(row_weight_sq, X.power(2)).ravel()/n_obs - \
			(1-q)*sparse.csc_matrix.dot(row_weight_sq, X).ravel()/n_obs
	
	mean = mm_M1
	variance = (mm_M2 - mm_M1**2)

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
		
	mm_M1 = (unique_expr*bootstrap_freq*inverse_size_factor).sum(axis=0)/n_obs
	mm_M2 = (unique_expr**2*bootstrap_freq*inverse_size_factor_sq - (1-q)*unique_expr*bootstrap_freq*inverse_size_factor_sq).sum(axis=0)/n_obs

	mean = mm_M1
	variance = (mm_M2 - mm_M1**2)

	return mean, variance


def compute_sev(
	X: sparse.csc_matrix,
	q: float,
	approx_size_factor: np.array,
	num_boot: int = 5000
):
	""" Compute the standard error of the variance. """
	
	n_obs = X.shape[0]
	inv_sf, inv_sf_sq, expr, counts = unique_expr(X, approx_size_factor)

	gen = np.random.Generator(np.random.PCG64(5))
	gene_rvs = gen.multinomial(n_obs, counts/counts.sum(), size=num_boot).T
	
	mean, var = compute_bootstrap_variance(
		unique_expr=expr,
		bootstrap_freq=gene_rvs,
		n_obs=n_obs,
		q=q,
		inverse_size_factor=inv_sf,
		inverse_size_factor_sq=inv_sf_sq
	)

	var = fill_invalid(var)

	sev = np.nanstd(var)
	selv = np.nanstd(np.log(var))

	return sev, selv
