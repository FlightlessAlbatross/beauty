#!/bin/bash


# Check if correct number of arguments is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input_vector_data> <amounts_raster> <output_raster>"
  echo $0
  echo $1
  echo $2
  echo $3
  exit 1
fi

# Set target bounding box and other constants
vectordata=$1
amounts_raster=$2
raster_output=$3
WEST=32280000
SOUTH=5235000
EAST=32922000
NORTH=6102000
EPSG_CODE="EPSG:4647"
PIXEL_SIZE=1000

# Execute the gdal_rasterize command
if [ ! -f "$amounts_raster" ]; then
  echo "Rasterizing OSM. Intermediary output saved to: $amounts_raster"
  gdal_rasterize -l points -burn 1 -a_nodata -99 -ot Int16 -te $WEST $SOUTH $EAST $NORTH -tr $PIXEL_SIZE $PIXEL_SIZE -init 0 -add $vectordata $amounts_raster
else
  echo "$amounts_raster already exists. Skipping generation."
fi

# Step 2: Find Maximum Value in the  Raster
MAX_RANGE=$(gdalinfo -mm $amounts_raster | grep "Computed Min/Max" | sed 's/.*Max=//' | awk '{print $1}')
echo "Maximum Range: $MAX_RANGE"

MIN_VALUE=$(echo $MAX_RANGE|awk -F, '{print $1}')
MAX_VALUE=$(echo $MAX_RANGE|awk -F, '{print $2}')

echo "Max value: $MAX_VALUE"


# Step 3: Normalize the Range to 0-1 by dividing by the maximum value
if [ ! -f "$raster_output" ]; then
  echo "Scaling OSM raster to 0,1"
  gdal_translate -ot Float64 -scale 0 $MAX_VALUE 0 1 $amounts_raster $raster_output 
else
    echo "$raster_output already exists. Skipping generation."
fi

