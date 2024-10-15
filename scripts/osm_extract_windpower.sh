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

# Execute ogr2ogr command to extract wind power plant data
ogr2ogr -f "GeoJSON" -t_srs $EPSG_CODE -where "other_tags LIKE '%\"power\"=>\"generator\"%' AND other_tags LIKE '%\"generator:source\"=>\"wind\"%'" "$output_file" "$input_file" points

echo "Wind power plant data extraction complete."
