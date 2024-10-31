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

# Directory for intermediary files
INTERMEDIARY_DIR=$(dirname "$INTERMEDIARY")
mkdir -p $INTERMEDIARY_DIR

# Define temporary files in the same directory as INTERMEDIARY
MAX_OUTFILE="$INTERMEDIARY_DIR/EU_DEM_EU_max.tif"
MIN_OUTFILE="$INTERMEDIARY_DIR/EU_DEM_EU_min.tif"

# Step 1: Generate Maximum Values Raster
if [ ! -f "$MAX_OUTFILE" ]; then
    echo "Generating maximum values raster..."
    gdalwarp -te $WEST $SOUTH $EAST $NORTH -te_srs $EPSG_CODE -tap -tr $PIXEL_SIZE $PIXEL_SIZE -t_srs $EPSG_CODE -ot $OUTPUT_TYPE -r max -co "TILED=$TILED" $INFILE $MAX_OUTFILE

else
    echo "$MAX_OUTFILE already exists. Skipping generation."
fi

# Step 2: Generate Minimum Values Raster
if [ ! -f "$MIN_OUTFILE" ]; then
    echo "Generating minimum values raster..."
    gdalwarp -te $WEST $SOUTH $EAST $NORTH -te_srs $EPSG_CODE -tap -tr $PIXEL_SIZE $PIXEL_SIZE -t_srs $EPSG_CODE -ot $OUTPUT_TYPE -r min -co "TILED=$TILED" $INFILE $MIN_OUTFILE

else
    echo "$MIN_OUTFILE already exists. Skipping generation."
fi


# Step 3: Calculate the Range
if [ ! -f "$INTERMEDIARY" ]; then
    echo "Generating difference between min and max values raster..."
    gdal_calc.py -A $MAX_OUTFILE -B $MIN_OUTFILE --outfile=$INTERMEDIARY --calc="A-B" --type=$OUTPUT_TYPE
else
    echo "$INTERMEDIARY already exists. Skipping generation."
fi

# Step 4: Find Maximum Value in the Range Raster
MAX_VALUE=$(gdalinfo -mm $INTERMEDIARY | grep "Computed Min/Max" | awk -F'[,=]' '{print $4}')


minmax=$(gdalinfo -mm $INTERMEDIARY|awk -F= '/Computed Min/ {print $2}')
MIN_RANGE=$(echo $minmax|awk -F, '{print $1}')
MAX_RANGE=$(echo $minmax|awk -F, '{print $2}')

echo "Maximum Range: $MAX_RANGE"




# Step 5: Normalize the Range to 0-1 by dividing by the maximum value
if [ ! -f "$RANGE_OUTFILE" ]; then
    echo "Generating scaled (0,1) output to $RANGE_OUTFILE"
    gdal_translate -scale 0 $MAX_RANGE 0 1 $INTERMEDIARY $RANGE_OUTFILE
else
    echo "$RANGE_OUTFILE already exists. Skipping generation."
fi


echo "Process completed. The final output is normalized between 0 and 1 at $RANGE_OUTFILE"
