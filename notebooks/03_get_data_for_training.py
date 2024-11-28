import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import os
from pathlib import Path
import json
import rasterio
from rasterio.mask import mask
import random

import argparse    # to make the script accept arguments. 
import importlib   # to dynamically import from the config file



from typing import List, Tuple


from rural_beauty.config import get_extracted_points_paths

from rural_beauty.config import (
    get_target_variable_raster_path,
    CLC_coverage_EU_dir,
    get_extracted_points_paths, 
    BFN_features_diverse, BFN_features_unique, BFN_features_beauty
)


def main(country: str, target_variable: str, sampling_method: str, overwrite:bool = True) -> None:
    '''
    This function extracts data from the UK or DE. 
    It consolidates multiple oder scripts that do similar things, 
    03_get_all_points_from_layers_DE, 03_get_points_from_layers_DE, 03_get_points_from_layers_UK

    It creates a folder in data / models / __extracted_points / {country} / {sampling} / 
    '''
    # setup path import library
    config_module = importlib.import_module("rural_beauty.config")
    
    # get output paths
    predictors_path, outcome_path, coords_path, feature_paths = get_extracted_points_paths(country, target_variable, sampling_method)

    # TODO: check if they exist. If yes and overwrite is True -> skip

    # get Boundary Polygon path
    NUTS_path = getattr(config_module, f"NUTS_{country}")

    # get feature data paths
    features = get_all_input_layer_paths()

    # get target variable data path
    outcome_raster = get_target_variable_raster_path(country, target_variable)

    # check path validity
    check_input_paths(features)

    # load Boundary Polygon
    NUTS = gpd.read_file(NUTS_path)
    polygon = NUTS.geometry.iloc[0]
    # add buffer to better extract coastal areas. 
    polygon = polygon.buffer(2500)    


    # extract outcome based on parameters target_variable and country/polygon
    match (country, sampling_method):
        case ('DE', 'all_pixels'):
            extraction_points, outcome_values = extract_all_raster_values_within_polygon(outcome_raster, polygon)
        case ('DE', 'random_pixels'):
            extraction_points, outcome_values = extract_random_raster_values_within_polygon(raster_path=outcome_raster, num_points= 10000, polygon=polygon)
            
        case ('UK', 'pooled_pixels_all_points'):
            # in the case of UK we don't extract a full raster, but only the points that have at least one image. 
            # those points are found in UK_scenic_points. 
            from rural_beauty.config import UK_scenic_points
            coords_gdf = gpd.read_file(UK_scenic_points)
            extraction_points = np.array([Point(x, y) for x, y in zip(coords_gdf.geometry.x, coords_gdf.geometry.y)])

            outcome_values = extract_raster_values_from_points(outcome_raster, extraction_points)
        case ("DEUK", _):
            raise ValueError('DEUK extraction is not yet implemented')
        case _:
            raise ValueError('unavailable options of country + sampling_method. Default is DE all_pixels or UK pooled_pixels_all_points' )

    # print(f"{type(extraction_points)}; length: {len(extraction_points)}")
    # format output objects:
    # we make sure we come out of the swich case with the data we need. 
    if 'extraction_points' in locals():
        coords_df = create_coords_df_from_extraction_points(extraction_points)
    else:
        raise ValueError(f"no extraction points object in locals for {country} {sampling_method}. This indicates a bug in the code!")
        
    
    # extract features for the same locations. 
    predictors_dict = {}
    for label, path in features.items():
        predictors_dict[label] = extract_raster_values_from_points(path, extraction_points)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(predictors_path), exist_ok=True)

    
    # outcome_values is a list, to ensure it is aligned with the predictors, which have headers, We add a generic header. 
    outcome_df = pd.DataFrame({'value': outcome_values})
    predictors_df = pd.DataFrame(predictors_dict)
    features_out = {k: str(v) for k, v in features.items()}


    # Write DataFrames to CSV files
    coords_df.to_csv(coords_path, index=False, float_format='%.15g')
    print(f"Coordinate file written to {coords_path}")
    outcome_df.to_csv(outcome_path, index=False, float_format='%.15g')
    print(f"Outcome file written to {outcome_path}")
    predictors_df.to_csv(predictors_path, index=False, float_format='%.15g')
    print(f"Predictors file written to {predictors_path}")

    with open(feature_paths, 'w') as output:
        json.dump(features_out, output, indent=4)
    print(f"Feature path json written to {feature_paths}")





def get_all_input_layer_paths() -> dict:
    # Combine all features from various sources
    features = {**BFN_features_diverse, **BFN_features_unique, **BFN_features_beauty}

    # Optionally include every CLC layer
    for file in os.listdir(CLC_coverage_EU_dir):
        if file.endswith(".tif"):
            temp_path = Path(CLC_coverage_EU_dir / file)
            temp_name = temp_path.stem[5:] + "_1"

            # Check if it is already in the features dict
            if temp_name in features.keys():
                continue
            else:
                features[temp_name] = temp_path

    return features

def check_input_paths(features) -> None:
    # Verify all files exist
    some_missing = False
    for label, path in features.items():
        if not os.path.exists(path):
            some_missing = True
            print(f"{path} is missing")

    if not some_missing:
        print("All files exist")


def extract_raster_values_from_points(raster_path, points) -> list[float]:
    # Function to extract raster values at given points and recode the NA value to -99
    with rasterio.open(raster_path) as src:
        nodata = src.nodata  # Get the nodata value from the raster metadata
        values = []
        for point in points:
            # Sample the raster at the given point
            sample = src.sample([(point.x, point.y)]).__next__()[0]
            if sample == nodata:
                values.append(-99)  # Recode nodata values to -1
            else:
                values.append(sample)  # Use the actual sample value if it's not nodata
    return values

def extract_all_raster_values_within_polygon(raster_path, polygon) -> Tuple[List[Point], list[float]]:
    with rasterio.open(raster_path) as src:
        # Mask the raster with the polygon
        out_image, out_transform = mask(src, [polygon], crop=True)
        nodata = src.nodata  # Get the NoData value

        # Extract valid pixel values and coordinates
        data = out_image[0]  # Extract the first band
        rows, cols = np.where(data != nodata)  # Indices of valid pixels
        values = data[rows, cols]  # Extract pixel values

        # Convert row/col indices to spatial coordinates
        coords = rasterio.transform.xy(out_transform, rows, cols)

        extraction_points = [Point(x, y) for x, y in zip(coords[0], coords[1])]
       
    return extraction_points, values


def create_coords_df_from_extraction_points(points):
    coords_dict = {}
    coords_dict['x_coord'] = [p.x for p in points]
    coords_dict['y_coord'] = [p.y for p in points]
    coords_df = pd.DataFrame(coords_dict)
    return coords_df


# Function to generate random points within a polygon
def generate_random_points(polygon, num_points) -> List[Point]:
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < num_points:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def extract_random_raster_values_within_polygon(raster_path, polygon, num_points) -> Tuple[List[Point], float]:
    random_points = generate_random_points(polygon, num_points)
    return random_points, extract_raster_values_from_points(raster_path, random_points)


def return_crs(raster_path):
    with rasterio.open(raster_path) as src:
        return src.crs

def return_NoDatavalue(raster_path):
    with rasterio.open(raster_path) as src:
        return src.nodata
    


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract data from raster files for specific countries and sampling methods.")
    parser.add_argument("country", type=str, help="Country code ('DE', 'UK', or 'DEUK').")
    parser.add_argument("target_variable", type=str, help="Target variable to extract (e.g., 'beauty', 'scenic').")
    parser.add_argument("sampling_method", type=str, help="Sampling method (e.g., 'all_pixels', 'random_pixels').")
    parser.add_argument("--overwrite", type=bool, default=True, help="Whether to overwrite existing data (default: True).")

    # Get arguments from command line
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(
        country=args.country,
        target_variable=args.target_variable,
        sampling_method=args.sampling_method,
        overwrite=args.overwrite
    )
