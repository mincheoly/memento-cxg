import concurrent
import logging
import multiprocessing
import sys
from concurrent import futures

import cell_census
import pandas as pd
import scipy.sparse
import scipy.sparse
import tiledbsoma as soma
from somacore import ExperimentAxisQuery, AxisQuery

from estimators import compute_mean, compute_sem, bin_size_factor, compute_variance, compute_sev

MIN_BATCH_SIZE = 1000

CUBE_DIMS_OBS = [
    "cell_type",
    "dataset_id",
]
CUBE_DIMS = ['feature_id'] + CUBE_DIMS_OBS

ESTIMATOR_NAMES = ['n', 'mean', 'sem', 'var', 'sev', 'selv']

Q = 0.1  # RNA capture efficiency depending on technology

MAX_WORKERS = None  # None means use multiprocessing's dynamic default

logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)


def compute_all_estimators(grouped):
    # TODO: transpose() is correct?
    X_single_gene = scipy.sparse.csc_matrix(grouped['soma_data']).transpose()
    approx_size_factor = grouped['approx_size_factor'].values

    n = X_single_gene.shape[0]  # sanity check; not required
    sample_mean, variance = compute_variance(X_single_gene, Q, approx_size_factor)
    mean = compute_mean(X_single_gene, Q, sample_mean, variance, approx_size_factor)
    sem = compute_sem(X_single_gene, variance)
    sev, selv = compute_sev(X_single_gene, Q, approx_size_factor, num_boot=10000)

    return pd.Series(data=[n, mean, sem, variance, sev, selv])


def compute_all_estimators_for_batch(soma_dim_0, obs_df: pd.DataFrame, var_df: pd.DataFrame, X_uri: str):
    with soma.SparseNDArray.open(X_uri) as X:
        X_df = X.read(coords=(soma_dim_0,
                              var_df.index.values)).tables().concat().to_pandas()
        logging.info(f"Pass 2: Processing X batch cells={len(soma_dim_0)}, nnz={len(X_df)}")
        # Compute estimators for each gene
        return (
            X_df.merge(var_df['feature_id'], left_on='soma_dim_1', right_index=True).
            merge(obs_df[CUBE_DIMS_OBS + ['approx_size_factor']], left_on='soma_dim_0', right_index=True).
            drop(columns=['soma_dim_0', 'soma_dim_1']).
            groupby(CUBE_DIMS, sort=False).
            apply(compute_all_estimators).
            rename(mapper=dict(enumerate(ESTIMATOR_NAMES)), axis=1)
        )


def sum_gene_expression_levels_by_cell(X_tbl: pa.Table) -> pd.Series:
    # TODO: use PyArrow API only; avoid Pandas conversion
    return X_tbl.to_pandas()[['soma_dim_0', 'soma_data']].groupby('soma_dim_0', sort=False).sum()['soma_data']


if __name__ == "__main__":
    census_soma = cell_census.open_soma(uri=sys.argv[1] if len(sys.argv) > 1 else None)

    organism_label = sys.argv[2] if len(sys.argv) > 2 else list(census_soma["census_data"].keys())[0]

    organism_census = census_soma["census_data"][organism_label]

    # init multiprocessing
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)
    ppe = futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)

    with ExperimentAxisQuery(organism_census,
                             measurement_name="RNA",
                             obs_query=AxisQuery(),  #value_filter="cell_type=='plasma cell'"),
                             var_query=AxisQuery(coords=(slice(0, 10000),))) as query:
        var_df = query.var().concat().to_pandas().set_index("soma_joinid")
        var_df['feature_id'] = var_df['feature_id'].astype('category')

        obs_df = (
            query.obs(column_names=["soma_joinid"] + CUBE_DIMS_OBS).
            concat().
            to_pandas().
            set_index("soma_joinid")
        )
        obs_df['dataset_id'] = obs_df['dataset_id'].astype('category')
        obs_df['cell_type'] = obs_df['cell_type'].astype('category')
        obs_df['size_factor'] = 0  # accumulated

        logging.info(f"Pass 1: Compute Approx Size Factors")

        futures = []
        for X_tbl in query.X("raw").tables():
            logging.info(f"Pass 1: Processing X batch nnz={X_tbl.shape[0]}")
            futures.append(ppe.submit(sum_gene_expression_levels_by_cell, X_tbl))

        for future in futures:
            # Accumulate cell sums, since a given cell's X values may be returned across multiple tables
            cell_sums = future.result()
            obs_df['size_factor'] = obs_df['size_factor'].add(cell_sums, fill_value=0)

        # Bin all sums to have fewer unique values, to speed up bootstrap computation
        obs_df['approx_size_factor'] = bin_size_factor(obs_df['size_factor'].values)

        logging.info(f"Pass 2: Compute Estimators")

        # accumulate into feature_id/cell_type/dataset_id Pandas multi-indexed DataFrame
        cube_index = pd.MultiIndex.from_arrays([[]] * 3, names=CUBE_DIMS)
        cube = pd.DataFrame(index=cube_index, columns=ESTIMATOR_NAMES)

        # Process X by cube rows. This ensures that estimators are computed
        # for all X data contributing to a given cube row aggregation.
        # TODO: `groups` converts categoricals to strs, which is inefficient
        cube_coords = obs_df[CUBE_DIMS_OBS].groupby(CUBE_DIMS_OBS).groups
        soma_dim_0_batch = []

        batch_futures = []

        for soma_dim_0_row in cube_coords.values():
            soma_dim_0_batch.extend(soma_dim_0_row)

            # Fetch data for multiple cube rows at once, to reduce X.read() call count
            if len(soma_dim_0_batch) < MIN_BATCH_SIZE:
                continue

            batch_futures.append(ppe.submit(compute_all_estimators_for_batch,
                                            soma_dim_0_batch,
                                            obs_df,
                                            var_df,
                                            query.experiment.ms['RNA'].X['raw'].uri))
            soma_dim_0_batch = []

        # Process final batch
        if len(soma_dim_0_batch) > 0:
            batch_futures.append(ppe.submit(compute_all_estimators_for_batch,
                                            soma_dim_0_batch,
                                            obs_df,
                                            var_df,
                                            query.experiment.ms['RNA'].X['raw'].uri))

        for n, future in enumerate(concurrent.futures.as_completed(batch_futures), start=1):
            result = future.result()
            cube = cube.append(result, ignore_index=True)
            logging.info(f"Pass 2: Completed [{n} of {len(batch_futures)}]")

        # TODO: Write to disk (e.g. as TileDB Array)
        print(cube)

