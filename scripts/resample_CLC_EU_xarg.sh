#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-overwrite] <input_directory> <output_directory>"
    exit 1
}

# Check for the correct number of arguments
if [ "$#" -lt 2 ]; then
    usage
fi

# Initialize variables
overwrite=0
if [ "$1" == "-overwrite" ]; then
    overwrite=1
    shift  # Shift the remaining arguments left
fi

# Now check if we have exactly two remaining arguments
if [ "$#" -ne 2 ]; then
    usage
fi

# Assign command line arguments to variables
DIR="$1"
OUT_DIR="$2"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Export variables for use in subshells (required for xargs)
export OUT_DIR
export overwrite


# Find all .tif files and process them with xargs
find "$DIR" -name "*.tif" | xargs -P 4 -I {} bash -c '
    raster="{}"
    out_raster="$OUT_DIR/$(basename "$raster")"

    # Check if the output file already exists and skip or overwrite based on the option provided
    if [ -f "$out_raster" ]; then
        if [ "$overwrite" -eq 0 ]; then
            echo "File $out_raster already exists, skipping..."
            exit 0
        fi
    fi

    # Run gdalwarp to resample the raster to 1000x1000 using mean aggregation
    # use -q to be quiet. 
    gdalwarp -q -te 28929820 3125857 35822750 8343990 -srcnodata 48 -dstnodata 48 -te_srs "EPSG:4647" -tap -tr 1000 1000 -t_srs "EPSG:4647" -ot float64 -r average $( [ "$overwrite" -eq 1 ] && echo "-overwrite" ) $raster $out_raster
'
echo CLC resampling to 1x1km grid complete.
