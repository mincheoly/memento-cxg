# memento
Generalized differential expression, including differential variability and coexpression


The `cell_census_summary_cube` script pre-computes the estimators that are used by Memento, using the CELLxGENE Census 
single-cell data. The estimators are output to a TileDB array named `estimators_cube`.

Usage instructions:
1. It is recommended to run this script on an AWS EC2 `r6id.24xlarge` instance in the `us-west-2` region. The instance must be configured with swap space, making use of the available SSD drives. Copy this [script](https://github.com/chanzuckerberg/cellxgene-census/blob/d9bd1eb4a3e14974a0e7d9c23fb8368e79b92c2d/tools/scripts/aws/swapon_instance_storage.sh) to the instance and run as root: `sudo swapon_instance_storage.sh`.
2. Setup a virtualenv and `pip install tiledbsoma`.
3. `git clone https://github.com/mincheoly/memento.git`
4. `git checkout atol/compute-estimators-on-cxg-census`
3. To run: `AWS_REGION=us-west-2 /usr/bin/time -v python ~/memento/memento/cell_census_summary_cube.py s3://cellxgene-data-public/cell-census/2023-04-04/soma/census_data/homo_sapiens 2>&1 | tee ~/memento/cell_census_summary_cube.log`

To use or inspect the results as Pandas DataFrame:
```
import tiledb
estimators = tiledb.open('estimators_cube').df[]:
```

Notes:
* The "size factors" are first computed for all cells (per cell) and stored in a TileDB Array called `obs_with_size_factor`. If the script is re-run, the size factors will be reloaded from this stored result. If you delete the `obs_with_size_factor` directory it will be recreated on the next run.
* The scripts makes use of Python's multiprocessing to parallelize the estimator computations. The amount of memory used per sub-process and overall on the instance will be impacted by the constants `MIN_BATCH_SIZE`, `TILEDB_SOMA_BUFFER_BYTES`, and `MAX_WORKERS`.
* The script takes ~12 hours to run in the default configuration on the `r6id.24xlarge` instance size.
