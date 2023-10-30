from datetime import datetime
import sys
import re

# List to store output lines
output_lines = []

# Dictionary to store start times and nnz values of batch processes
start_times = {}

# Dictionary to store the maximum nnz value observed by each batch process during its execution
max_nnz_active = {}

cells_active = nnz_active = 0

for line in sys.stdin:
    parts = line.split()
    if len(parts) < 11:
        continue

    timestamp = " ".join(parts[:2])

    # Use regular expressions to extract the batch ID and nnz value
    batch_match = re.search(r'batch (\d+)', line)
    nnz_match = re.search(r'nnz=(\d+)', line)
    cells_match = re.search(r'cells=(\d+)', line)

    if batch_match:
        batch_id = int(batch_match.group(1))
    else:
        continue

    if "Start" in line or "End" in line:
        cells = int(cells_match.group(1)) if cells_match else None
        nnz = int(nnz_match.group(1)) if nnz_match else None

    if "Start" in line:
        # Store the start time and nnz value (if available)
        start_times[batch_id] = {"start_time": datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"), "nnz_value": nnz_match.group(1) if nnz_match else None}
        cells_active += cells
        nnz_active += nnz
        for batch_id in max_nnz_active.keys():
            max_nnz_active[batch_id] = max(max_nnz_active[batch_id], nnz_active)
        max_nnz_active[batch_id] = nnz_active

    elif "End" in line:
        if batch_id in start_times:
            end_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            elapsed_time = end_time - start_times[batch_id]["start_time"]

            # Calculate rates
            nnz_per_second = nnz / elapsed_time.total_seconds() if nnz else None
            cells_per_second = cells / elapsed_time.total_seconds() if cells else None

            # Store the output line in the list
            output_line = (f"Batch {batch_id}: Elapsed time: {elapsed_time}, "
                           f"cells: {cells}, cells/sec: {cells_per_second:.2f}, "
                           f"nnz: {(nnz / 1000000):.2f}M, nnz/sec: {nnz_per_second:.2f}, "
                           f"max_nnz_active: {(max_nnz_active[batch_id] / 1000000):.2f}M ")
            output_lines.append((elapsed_time, output_line))
            del start_times[batch_id]
            del max_nnz_active[batch_id]

# Sort the output lines by elapsed time (lowest first)
output_lines.sort()

# Print the sorted output
for _, output_line in output_lines:
    print(output_line)
