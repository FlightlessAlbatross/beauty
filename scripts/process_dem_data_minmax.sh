#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input_elevation_data> <intermediary_range_raster> <scaled_range_raster>"
  echo "Check if the arguments are empty strings."
  exit 1
fi

# Arguments
INFILE=$1
INTERMEDIARY=$2
RANGE_OUTFILE=$3

# Set target bounding box and other constants
WEST=32280000
SOUTH=5235000
EAST=32922000
NORTH=6102000
EPSG_CODE="EPSG:4647"
PIXEL_SIZE=1000
OUTPUT_TYPE="Float64"
COMPRESS="LZW"
PREDICTOR=2
TILED="YES"

# Directory for intermediary files
INTERMEDIARY_DIR=$(dirname "$INTERMEDIARY")
mkdir -p $INTERMEDIARY_DIR

# Define temporary files in the same directory as INTERMEDIARY
MAX_OUTFILE="$INTERMEDIARY_DIR/__deletable_EU_DEM_EU_max.tif"
MIN_OUTFILE="$INTERMEDIARY_DIR/__deletable_EU_DEM_EU_min.tif"

echo "This script produces 2 intermediary files with the prefix __deletable_ in the current directory. They get removed by this script."

# Step 1: Generate Maximum Values Raster
gdalwarp -te $WEST $SOUTH $EAST $NORTH -te_srs $EPSG_CODE -tap -tr $PIXEL_SIZE $PIXEL_SIZE -t_srs $EPSG_CODE -ot $OUTPUT_TYPE -r max -co "TILED=$TILED" $INFILE $MAX_OUTFILE

# Step 2: Generate Minimum Values Raster
gdalwarp -te $WEST $SOUTH $EAST $NORTH -te_srs $EPSG_CODE -tap -tr $PIXEL_SIZE $PIXEL_SIZE -t_srs $EPSG_CODE -ot $OUTPUT_TYPE -r min -co "TILED=$TILED" $INFILE $MIN_OUTFILE

# Step 3: Calculate the Range (Max - Min)
gdal_calc.py -A $MAX_OUTFILE -B $MIN_OUTFILE --outfile=$INTERMEDIARY --calc="A-B" --type=$OUTPUT_TYPE

# Step 4: Find Maximum Value in the Range Raster
MAX_VALUE=$(gdalinfo -mm $INTERMEDIARY | grep "Computed Min/Max" | sed 's/.*Max=//' | awk '{print $1}')

# Step 5: Normalize the Range to 0-1 by dividing by the maximum value
gdal_calc.py -A $INTERMEDIARY --outfile=$RANGE_OUTFILE --calc="A/$MAX_VALUE" --type=$OUTPUT_TYPE

# Clean up temporary files
rm $MIN_OUTFILE
rm $MAX_OUTFILE


echo "Process completed. The final output is normalized between 0 and 1."
