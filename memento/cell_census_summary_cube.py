import itertools
import scipy.sparse
import sys

import cell_census
import pandas as pd
import scipy.sparse
import somacore
from somacore import ExperimentAxisQuery, AxisQuery

from estimators import compute_mean, compute_sem, bin_size_factor, compute_variance, compute_sev

cube_dims_obs = [
    "cell_type",
    "dataset_id",
]
cube_dims = ["gene_ontology_term_id"] + cube_dims_obs

estimator_names = ['mean', 'sem']

Q = 0.1  # RNA capture efficiency depending on technology


def compute_all_estimators(grouped):
    # TODO: transpose() is correct?
    X_single_gene = scipy.sparse.csc_matrix(grouped['soma_data']).transpose()
    approx_size_factor = grouped['approx_size_factor'].values

    sample_mean, variance = compute_variance(X_single_gene, Q, approx_size_factor)
    mean = compute_mean(X_single_gene, Q, sample_mean, variance, approx_size_factor)
    sem = compute_sem(X_single_gene, variance)
    sev, selv = compute_sev(X_single_gene, Q, approx_size_factor, num_boot=10000)

    return mean, sem, variance, sev, selv


def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


if __name__ == "__main__":
    census_soma = cell_census.open_soma(uri=sys.argv[1] if len(sys.argv) > 1 else None)

    organism_label = sys.argv[2] if len(sys.argv) > 2 else list(census_soma["census_data"].keys())[0]

    organism_census = census_soma["census_data"][organism_label]

    with ExperimentAxisQuery(organism_census, measurement_name="RNA",
                             obs_query=AxisQuery()) as query:
        var_df = query.var().concat().to_pandas().set_index("soma_joinid")
        obs_df = query.obs(column_names=["soma_joinid"] + cube_dims_obs).concat().to_pandas().\
            set_index("soma_joinid")

        var_df['feature_id'] = var_df['feature_id'].astype('category')
        obs_df['dataset_id'] = obs_df['dataset_id'].astype('category')
        obs_df['cell_type'] = obs_df['cell_type'].astype('category')
        obs_df['size_factor'] = 0

        print(f"Pass 1: Compute Approx Size Factors")

        for X_tbl in query.X("raw").tables():
            print(f"Pass 1: Processing X batch size={X_tbl.shape[0]}")
            X_df = X_tbl.to_pandas()

            # Sum expression levels for each cell
            cell_sums = (
                X_df.groupby('soma_dim_0', sort=False).
                aggregate(sum=pd.NamedAgg("soma_data", "sum"))
            )

            # Accumulate cell sums, since a given cell's X values may be returned across multiple tables
            obs_df['size_factor'] = obs_df['size_factor'].add(cell_sums['sum'])

        # Bin all sums to have fewer unique values, to speed up bootstrap computation
        obs_df['approx_size_factor'] = bin_size_factor(obs_df['size_factor'].values)

        print(f"Pass 2: Compute Estimators")

        # accumulate into feature_id/cell_type/dataset_id Pandas multi-indexed DataFrame
        # cube_index = pd.MultiIndex.from_arrays([[]] * 3, names=['feature_id'] + cube_dims_obs)
        # cube = pd.DataFrame(index=cube_index, columns=estimator_names)
        estimators = {}

        # Process X by cube rows
        X: somacore.SparseNDArray = query.experiment.ms['RNA'].X['raw']
        cube_coords = obs_df[cube_dims_obs].groupby(cube_dims_obs).groups
        for group_key, soma_dim_0_batch in cube_coords.items():
            X_df = X.read(coords=(soma_dim_0_batch.to_numpy(),)).tables().concat().to_pandas()

            print(f"Pass 2: processing X batch size={X_df.shape[0]}, cells={X_df['soma_dim_0'].nunique()}")

            # cube_batch_result = (
            #     X_df.merge(var_df['feature_id'], left_on='soma_dim_1', right_index=True).
            #     merge(obs_df[cube_dims_obs + ['approx_size_factor']], left_on='soma_dim_0', right_index=True).
            #     drop(columns=['soma_dim_0', 'soma_dim_1']).
            #     groupby(['feature_id'] + cube_dims_obs, sort=False).
            #     apply(compute_all_estimators).
            #     rename(mapper=dict(enumerate(estimator_names)), axis=1)
            # )
            # cube = pd.concat([cube, cube_batch_result])

            estimators[group_key] = compute_all_estimators(
                X_df.set_index('soma_dim_0').join(obs_df[['approx_size_factor']]))

        # print(cube)
        print(estimators)
