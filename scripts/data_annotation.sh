#!/bin/bash
start_idx=$1
end_idx=$2
DEVICE=$3
echo "start_idx=$start_idx, end_idx=$end_idx"
num_processes=25
step=$(( (end_idx - start_idx) / num_processes ))

for ((i=0; i<num_processes; i++)); do
    i_start=$(( start_idx + i * step ))
    i_end=$(( start_idx + (i + 1) * step ))

    # Ensure last process gets the full range
    if [ $i -eq $((num_processes - 1)) ]; then
        i_end=$end_idx
    fi

    echo "Running process $i: i_start=$i_start, i_end=$i_end"

    # Run in background
    # CUDA_VISIBLE_DEVICES=$DEVICE python -m src.dataset.annotate_data --path ./data/clip-filtered-dataset --i_start $i_start --i_end $i_end &
    CUDA_VISIBLE_DEVICES=$DEVICE python annotate_data.py --path datasets/laion400m-data --i_start $i_start --i_end $i_end &
done

# Wait for all background processes to finish
wait

echo "All processes completed."
