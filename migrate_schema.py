import sys

import tiledb

from memento.cell_census_summary_cube import CUBE_SCHEMA as NEW_CUBE_SCHEMA

if __name__ == '__main__':
    old_cube_uri = sys.argv[1]
    new_cube_uri = sys.argv[2]

    tiledb.Array.create(new_cube_uri, NEW_CUBE_SCHEMA, overwrite=False)

    with tiledb.open(old_cube_uri, 'r') as old_cube, tiledb.open(new_cube_uri, 'w') as new_cube:
        for i, old_chunk in enumerate(old_cube.query(return_incomplete=True, use_arrow=True, return_arrow=True).df[:], start=1):
            print(f"writing chunk {i}, shape={old_chunk.shape}")
            coords = [old_chunk[dim.name].combine_chunks() for dim in old_cube.schema.domain]
            data = {attr.name: old_chunk[attr.name].combine_chunks() for attr in old_cube.schema}
            new_cube[tuple(coords)] = data





