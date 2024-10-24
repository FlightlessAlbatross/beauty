#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <path_to_shp0> <path_to_shp1> <path_to_shp2> <output_tif>"
    exit 1
fi

# Input arguments
shp0="$1"
shp1="$2"
shp2="$3"
output_tif="$4"

output_epsg=4647

# Define output paths for intermediate results
subset0="data/processed/protected/WDPA0.shp"
subset1="data/processed/protected/WDPA1.shp"
subset2="data/processed/protected/WDPA2.shp"
merged="data/processed/protected/WDPA_merged.shp"
final_output="data/processed/protected/WDPA_merged_4647.shp"

# Define bounding box (xmin, ymin, xmax, ymax)
xmin=-24.5
ymin=31
xmax=33
ymax=69

# Step 1: Subset the shapefiles using the bounding box and filter by MARINE != 2
ogr2ogr -f "ESRI Shapefile" $subset0 $shp0 -spat $xmin $ymin $xmax $ymax -where "MARINE <> '2'"
ogr2ogr -f "ESRI Shapefile" $subset1 $shp1 -spat $xmin $ymin $xmax $ymax -where "MARINE <> '2'"
ogr2ogr -f "ESRI Shapefile" $subset2 $shp2 -spat $xmin $ymin $xmax $ymax -where "MARINE <> '2'"

# Step 2: Merge the three subsets into a single shapefile
ogr2ogr -f "ESRI Shapefile" $merged $subset0
ogr2ogr -f "ESRI Shapefile" -update -append $merged $subset1 -nln merged_shp
ogr2ogr -f "ESRI Shapefile" -update -append $merged $subset2 -nln merged_shp

# Step 3: Reproject the merged shapefile to EPSG:4647
ogr2ogr -f "ESRI Shapefile" -t_srs "EPSG:4647" $final_output $merged

# Clean up intermediate files
rm -f $subset0 $subset1 $subset2
