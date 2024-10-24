#!/bin/bash

# Arguments
INFILE=$1
RANGE_OUTFILE=$2

# Set target bounding box and other constants
#WEST=9421250
#SOUTH=9417500
#EAST=65258750
#NORTH=64056750
WEST=29172876.762312
EAST=35737830.291022
SOUTH=3927473.808968
NORTH=7986389.044350
EPSG_CODE="EPSG:4647"
PIXEL_SIZE=1000
OUTPUT_TYPE="Float64"
COMPRESS="LZW"
PREDICTOR=2
TILED="YES"

# Temporary files
MAX_OUTFILE="__deletable_EU_DEM_EU_max.tif"
MIN_OUTFILE="__deletable_EU_DEM_EU_min.tif"

echo This script produces 2 intermediary files with the prefix __deletable_ in the current directory. They get removed by this script. 
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
