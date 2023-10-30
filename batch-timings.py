from datetime import datetime
import sys
import re

# List to store output lines
output_lines = []

# Dictionary to store start times and nnz values of batch processes
start_times = {}

for line in sys.stdin:
    parts = line.split()
    if len(parts) < 11:
        continue

    timestamp = " ".join(parts[:2])

    # Use regular expressions to extract the batch ID and nnz value
    batch_match = re.search(r'batch (\d+)', line)
    nnz_match = re.search(r'nnz=(\d+)', line)

    if batch_match:
        batch_id = int(batch_match.group(1))
    else:
        continue

    if "Start" in line:
        # Store the start time and nnz value (if available)
        start_times[batch_id] = {"start_time": datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"), "nnz_value": nnz_match.group(1) if nnz_match else None}

    elif "End" in line:
        if batch_id in start_times:
            end_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            elapsed_time = end_time - start_times[batch_id]["start_time"]
            nnz_value = int(nnz_match.group(1)) if nnz_match else None

            # Calculate nnz per second
            nnz_per_second = nnz_value / elapsed_time.total_seconds() if nnz_value else None

            # Store the output line in the list
            output_line = f"Batch {batch_id}: Elapsed time: {elapsed_time}, nnz: {nnz_value // 1000000}M, nnz per second: {nnz_per_second:.2f}"
            output_lines.append((elapsed_time, output_line))
            del start_times[batch_id]

# Sort the output lines by elapsed time (lowest first)
output_lines.sort()

# Print the sorted output
for _, output_line in output_lines:
    print(output_line)
