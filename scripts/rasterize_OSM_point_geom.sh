#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_vector_data> <output_raster>"
  exit 1
fi

# Set target bounding box and other constants
vectordata=$1
raster_output=$2
WEST=32280000
SOUTH=5235000
EAST=32922000
NORTH=6102000
EPSG_CODE="EPSG:4647"
PIXEL_SIZE=1000

# Execute the gdal_rasterize command
gdal_rasterize -l points -burn 1 -a_nodata -99 -ot Int16 -te $WEST $SOUTH $EAST $NORTH -tr $PIXEL_SIZE $PIXEL_SIZE -init 0 -add $vectordata $raster_output

echo "Rasterization complete. Output saved to: $raster_output"
