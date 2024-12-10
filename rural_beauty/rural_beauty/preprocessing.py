# in this script we go from the raw data to datasets for training and predicting. 

import os
import subprocess

import argparse

import json
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, mapping
from scipy.spatial import cKDTree   # To interpoloate the UK data to all pixels. 
from rasterio.transform import from_origin
from rasterio.features import geometry_mask

from concurrent.futures import ThreadPoolExecutor


# Beauty/Uniqueness/Diversity

def main(skip_DE= False,
         skip_UK= False,
         skip_DEM= False,
         skip_OSM= False,
         skip_CLC= False,
         skip_Hemerobie= False,
         skip_Protected= False,
         skip_Neighborhood= False):
    

    if skip_DE:
        print("Skipping Preprocessing of German Beauty/Scenic/Diversity")
    # import paths to the German Beauty dataset and the output for the rasterization
    else:
        from rural_beauty.config import bild_vector_dir, bild_raster_dir # load absolute paths as defined in config.py
        rasterize_DE_outcome(bild_vector_dir, bild_raster_dir)


    if skip_UK:
        print("Skipping Preprocessing of UK Scenic")
    else:
        # UK Scenic data
        from rural_beauty.config import UK_scenic_raw, UK_scenic_points, UK_scenic_raster, NUTS_UK
        UK_gdf = make_scenic_geojson(UK_scenic_raw)
        UK_gdf.to_file(UK_scenic_points)
        rasterize_UK_outcome(UK_gdf, UK_scenic_raster, NUTS_UK)


    if skip_DEM:
        print("Skipping Preprocessing of the Digital Elevation Model")
    else:
        # Digital Elevation model
        from rural_beauty.config import DEM_EU, DEM_EU_range, DEM_EU_range_scaled
        process_DEM(DEM_EU, DEM_EU_range, DEM_EU_range_scaled)


    if skip_OSM:
        print("Skipping Preprocessing of the Open Streetmap Data")
    else:
        # OSM data
        from rural_beauty.config import OSM_full_EU
        from rural_beauty.config import powerlines_EU_vector, powerlines_EU_raster, powerlines_EU_raster_scaled
        from rural_beauty.config import streets_EU_vector, streets_EU_raster, streets_EU_raster_scaled
        from rural_beauty.config import windpower_EU_vector, windpower_EU_raster, windpower_EU_raster_scaled

        process_OSM_all(OSM_full_EU, 
                        powerlines_EU_vector, powerlines_EU_raster, powerlines_EU_raster_scaled, 
                        streets_EU_vector, streets_EU_raster, streets_EU_raster_scaled, 
                        windpower_EU_vector, windpower_EU_raster, windpower_EU_raster_scaled)


    if skip_CLC:
        print("Skipping Preprocessing of Corine Landcover")
    else:
        # clc
        from rural_beauty.config import CLC_EU, CLC_boolean_layers_dir, CLC_coverage_EU_dir
        from rural_beauty import split_CLC_layers_multithread as split_CLC_layers

        # split landcover data into separate tifs of "share of Landcover Class X per pixel"
        split_CLC_layers.main(CLC_EU, CLC_boolean_layers_dir)


        # rasterize to EU extent and 1kmx1km grid. 
        subprocess.run(["bash", "scripts/resample_CLC_EU_xarg.sh", 
                        CLC_boolean_layers_dir, CLC_coverage_EU_dir])


    if skip_Hemerobie:
        print("Skipping Preprocessing of the Hemerobie index data")
    else:
        # Hemerobieindex
        from rural_beauty.config import heme2012_DE, heme2012_DE_repojected
        subprocess.run(["bash", "scripts/reproject_heme.sh", 
                        heme2012_DE, heme2012_DE_repojected])

    if skip_Protected:
        print("Skipping Preprocessing of the Protected areas")
    else:
        # Protected Areas
        from rural_beauty.config import protected0, protected1, protected2, protected_EU, protected_raster, protected_raster_scaled

        subprocess.run(["bash", "scripts/WDPA_subset_reproject_xarg.sh", 
                    protected0, protected1, protected2, protected_EU])
        subprocess.run(["Rscript", "scripts/rasterize_protected_poly_geom_EU.R", 
                    protected_EU, protected_raster, protected_raster_scaled])
    
    if skip_Neighborhood:
        print("Skipping Creation of Neighborhood Layers")
    else:
        # Neighborhood Values
        from rural_beauty import neighborhood_values_multithread as neighborhood_values
        neighborhood_values.main()






def process_OSM_all(OSM_full_EU, 
                    powerlines_EU_vector, powerlines_EU_raster, powerlines_EU_raster_scaled,
                    streets_EU_vector, streets_EU_raster, streets_EU_raster_scaled,
                    windpower_EU_vector, windpower_EU_raster, windpower_EU_raster_scaled):
    """
    Process OSM data by running a series of shell and R scripts for powerlines, streets, and windpower.
    """

    # Function to run a single subprocess. used for multithreding 
    def run_subprocess(command):
        return subprocess.run(command)

    commands_extract_osm = [
    ['bash', 'scripts/osm_extract_streets.sh'   , OSM_full_EU, streets_EU_vector],
    ['bash', 'scripts/osm_extract_powerlines.sh', OSM_full_EU, powerlines_EU_vector],
    ['bash', 'scripts/osm_extract_windpower.sh' ,  OSM_full_EU, windpower_EU_vector]
    ]

    commands_rasterize_osm = [
        ['Rscript', 'scripts/rasterize_OSM_line_geom_EU.R', streets_EU_vector, streets_EU_raster, streets_EU_raster_scaled], 
        ['Rscript', 'scripts/rasterize_OSM_line_geom_EU.R', powerlines_EU_vector, powerlines_EU_raster, powerlines_EU_raster_scaled], 
        ['bash', 'scripts/rasterize_OSM_point_geom_EU.sh', windpower_EU_vector, windpower_EU_raster, windpower_EU_raster_scaled]
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:  # Run up to 3 tasks concurrently
        futures = [executor.submit(run_subprocess, cmd) for cmd in commands_extract_osm]
        # Wait for all tasks to complete
        for future in futures:
            future.result()

    with ThreadPoolExecutor(max_workers=3) as executor:  # Run up to 3 tasks concurrently
        futures = [executor.submit(run_subprocess, cmd) for cmd in commands_rasterize_osm]
        # Wait for all tasks to complete
        for future in futures:
            future.result()


def make_scenic_geojson(UK_scenic_raw):
    # Load the TSV file into a DataFrame
    df = pd.read_csv(UK_scenic_raw, sep='\t')

    # Create a geometry column using Lat and Lon for points
    df['geometry'] = df.apply(lambda row: Point(row['Lon'], row['Lat']), axis=1)

    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Set a Coordinate Reference System (CRS) if needed (e.g., WGS84)
    gdf.set_crs("EPSG:4326", inplace=True)

    # transform it to 4346 so we can use meters based grids. 
    gdf.to_crs("EPSG:4647", inplace=True)

    return gdf


def process_DEM(DEM_EU, DEM_EU_range, DEM_EU_range_scaled) -> None:
   subprocess.run(["bash", "scripts/process_dem_data_minmax_EU.sh",
                   DEM_EU, DEM_EU_range, DEM_EU_range_scaled])


def rasterize_DE_outcome(bild_vector_dir, bild_raster_dir):
# Take every .shp file in the -i directory and rasterizes it into the -o direcory and adjusts the naming. 
    subprocess.run(["Rscript",
                    "scripts/rasterize_ratings.R",
                    '-i', bild_vector_dir,
                    '-o', bild_raster_dir])

    print(f"German Beauty/Uniqueness/Diversity rasterized and saved to: {bild_raster_dir}")


def rasterize_UK_outcome(points_gdf, UK_scenic_raster, NUTS_UK) -> None:
    # Load point data with values
    gdf = points_gdf
    points = np.array([(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)])
    values = gdf['Average'].to_numpy()  # Replace with your actual column

    # Define the raster grid with desired extent and resolution
    minx, miny, maxx, maxy = gdf.total_bounds
    resolution = 1000  # 1 km resolution
    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(miny, maxy, resolution)
    xi, yi = np.meshgrid(x_coords, y_coords)

    # Interpolate values using IDW
    interpolated_values = idw_interpolation(points, values, xi, yi)

    # mask values outside the UK
    gdf_boundary = gpd.read_file(NUTS_UK)
    # Create the mask using geometries instead of shapes
    transform = from_origin(minx, maxy, resolution, resolution)
    mask = geometry_mask(
        geometries=gdf_boundary.geometry.buffer(3000),  # Note: 'geometries' is used here instead of 'shapes'
        out_shape=xi.shape,
        transform=transform,
        invert=True  # Set cells inside geometries to True, outside to False
    )


    flipped_values = np.flipud(interpolated_values)

    nodata_value = -99
    masked_values = np.where(mask, flipped_values, nodata_value)
    
    # Write the interpolated raster to a GeoTIFF file
    transform = from_origin(minx, maxy, resolution, resolution)
    with rasterio.open(
        UK_scenic_raster,
        "w",
        driver="GTiff",
        height=interpolated_values.shape[0],
        width=interpolated_values.shape[1],
        count=1,
        nodata = -99,
        dtype='float32',
        crs=gdf.crs,
        transform=transform
    ) as dst:
        dst.write(masked_values.astype('float32'), 1)

    print(f"Rasterization of UK done. Written to {UK_scenic_raster}")


def idw_interpolation(points, values, xi, yi, power=2):
    # Define IDW function
    tree = cKDTree(points)
    dist, idx = tree.query(np.array([xi.ravel(), yi.ravel()]).T, k=5)  # Use the 5 nearest neighbors
    weights = 1 / dist**power
    weights[dist == 0] = 0  # Avoid division by zero for exact points
    z = np.sum(weights * values[idx], axis=1) / np.sum(weights, axis=1)
    return z.reshape(xi.shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process various datasets.")
    parser.add_argument('--skip_DE'    , action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_UK'    , action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_DEM'           , action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_OSM'           , action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_CLC'           , action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_Hemerobie'     , action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_Protected'     , action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip_Neighborhood'  , action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(
        skip_DE          =args.skip_DE,
        skip_UK          =args.skip_UK,
        skip_DEM         =args.skip_DEM,
        skip_OSM         =args.skip_OSM,
        skip_CLC         =args.skip_CLC,
        skip_Hemerobie   =args.skip_Hemerobie,
        skip_Protected   =args.skip_Protected,
        skip_Neighborhood=args.skip_Neighborhood
        )