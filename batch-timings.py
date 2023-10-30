from datetime import datetime
import sys
import re

# List to store output lines
output_lines = []

# Dictionary to store start times and nnz values of batch processes
batch_stats = {}

cells_active = nnz_active = 0

for line in sys.stdin:
    # Use regular expressions to extract relevant fields
    timestamp = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
    batch_match = re.search(r'batch (\d+)', line)
    nnz_match = re.search(r'nnz=(\d+)', line)
    cells_match = re.search(r'cells=(\d+)', line)

    if batch_match:
        batch_id = int(batch_match.group(1))
    else:
        continue

    if "Start" in line:
        cells = int(cells_match.group(1)) if cells_match else 0
        nnz = int(nnz_match.group(1)) if nnz_match else 0

        cells_active += cells
        nnz_active += nnz

        for batch_id in batch_stats.keys():
            batch_stats[batch_id]["max_nnz"] = max(batch_stats[batch_id]["max_nnz"], nnz_active)

        # Store the start time and nnz value (if available)
        batch_stats[batch_id] = {"start_time": datetime.strptime(timestamp.group(1), "%Y-%m-%d %H:%M:%S"),
                                 "nnz": nnz,
                                 "cells": cells,
                                 "max_nnz": nnz_active}

    elif "End" in line:
        if batch_id in batch_stats:
            end_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            elapsed_time = end_time - batch_stats[batch_id]["start_time"]

            cells = batch_stats[batch_id]["cells"]
            nnz = batch_stats[batch_id]["nnz"]
            max_nnz = batch_stats[batch_id]['max_nnz']

            cells_active -= cells
            nnz_active -= nnz

            # Calculate rates
            nnz_per_second = nnz / elapsed_time.total_seconds() if nnz else None
            cells_per_second = cells / elapsed_time.total_seconds() if cells else None

            # Store the output line in the list
            output_line = (f"Batch {batch_id}: Elapsed time: {elapsed_time}, "
                           f"cells: {cells}, cells/sec: {cells_per_second:.2f}, "
                           f"nnz: {(nnz / 1000000):.2f}M, nnz/sec: {nnz_per_second:.2f}, "
                           f"max_nnz_active: {(max_nnz / 1000000):.2f}M ")
            output_lines.append((elapsed_time, output_line))
            del batch_stats[batch_id]

# Sort the output lines by elapsed time (lowest first)
output_lines.sort()

# Print the sorted output
for _, output_line in output_lines:
    print(output_line)
