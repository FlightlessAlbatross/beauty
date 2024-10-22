#!/bin/bash

#TODO: all the paths are hardcoded. This should be incorporated into the config scheme. 

# Define input shapefiles
shp0="data/raw/protected/WDPA_Oct2024_Public_shp_0/WDPA_Oct2024_Public_shp-polygons.shp"
shp1="data/raw/protected/WDPA_Oct2024_Public_shp_1/WDPA_Oct2024_Public_shp-polygons.shp"
shp2="data/raw/protected/WDPA_Oct2024_Public_shp_2/WDPA_Oct2024_Public_shp-polygons.shp"

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
