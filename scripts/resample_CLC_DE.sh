#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Assign command line arguments to variables
DIR="$1"
OUT_DIR="$2"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Loop through all TIFF files in the directory
for raster in "$DIR"/*.tif; do
    # Define output file path
    out_raster="$OUT_DIR"/$(basename "$raster")
    
    # Run gdalwarp to resample the raster to 1000x1000 using mean aggregation
    gdalwarp -te 32280000 5235000 32922000 6102000 -te_srs "EPSG:4647" -tap -tr 1000 1000 -t_srs "EPSG:4647" -ot float64 -r average "$raster" "$out_raster"
done

echo "Resampling complete."