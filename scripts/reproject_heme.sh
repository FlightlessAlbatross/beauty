#!/bin/bash

# this script reprojects an input tif to german extents in EPSG:4647 to a 1km x 1km grid by averaging the input pixels. 

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Assign command line arguments to variables
input_file="$1"
output_file="$2"

# create the output folder
mkdir -p "$(dirname "$output_file")"

# run the process
gdalwarp -te 32280000 5235000 32922000 6102000 -te_srs "EPSG:4647" -tap -tr 1000 1000 -t_srs "EPSG:4647" -ot float64 -r average $input_file $output_file
