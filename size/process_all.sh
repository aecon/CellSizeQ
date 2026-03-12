#!/usr/bin/env bash
set -euo pipefail


# Check arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path/to/directory/containing/HA_xx>"
    exit 1
fi

inputdir0="$1"


# Check if input dir exists
if [[ ! -d "$inputdir0" ]]; then
    echo "Error: directory '$inputdir0' does not exist"
    exit 1
fi


# Make output folder
mkdir -p results


# Find directories
shopt -s nullglob # changes how filename globbing behaves when no files match
for dir in "$inputdir0"/HA*; do
    [[ -d "$dir" ]] || continue

    echo "Processing: $dir"

    platename="$(basename "$dir")"
    echo "Plate name: $platename"

    python process_FastParallel_hestia_cell_and_nuclei_area_diameter.py \
        -n "$platename" \
        -d "$dir"
done

echo "Done."

