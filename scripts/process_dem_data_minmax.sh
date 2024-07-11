#!/bin/bash

# Arguments
INFILE=$1
RANGE_OUTFILE=$2

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

# Temporary files
MAX_OUTFILE="EU_DEM_DE_max.tif"
MIN_OUTFILE="EU_DEM_DE_min.tif"

# Step 1: Generate Maximum Values Raster
gdalwarp -te $WEST $SOUTH $EAST $NORTH -te_srs $EPSG_CODE -tap -tr $PIXEL_SIZE $PIXEL_SIZE -t_srs $EPSG_CODE -ot $OUTPUT_TYPE -r max -co "TILED=$TILED" $INFILE $MAX_OUTFILE

# Step 2: Generate Minimum Values Raster
gdalwarp -te $WEST $SOUTH $EAST $NORTH -te_srs $EPSG_CODE -tap -tr $PIXEL_SIZE $PIXEL_SIZE -t_srs $EPSG_CODE -ot $OUTPUT_TYPE -r min -co "TILED=$TILED" $INFILE $MIN_OUTFILE

# Step 3: Calculate the Range (Max - Min)
gdal_calc.py -A $MAX_OUTFILE -B $MIN_OUTFILE --outfile=$RANGE_OUTFILE --calc="A-B" --type=$OUTPUT_TYPE

# Clean up temporary files
rm $MIN_OUTFILE
rm $MAX_OUTFILE

echo "Process completed."
