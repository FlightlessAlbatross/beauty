#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
}

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    usage
fi

# Assign command line arguments to variables
input_file="$1"
output_file="$2"

# Set EPSG code
EPSG_CODE="EPSG:4647"

echo "Extract street vectors from OSM Geofabrik file..."

# Execute ogr2ogr command to extract street data
ogr2ogr -f "GeoJSON" -t_srs $EPSG_CODE -where "highway='trunk' OR highway='motorway' OR highway='primary' OR highway='secondary'" "$output_file" "$input_file" lines

echo "Street data extraction complete."
