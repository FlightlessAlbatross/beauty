

import numpy as np
import pandas as pd
import geopandas as gpd
import json
import os
from pathlib import Path
import argparse
# import importlib

# various ML models. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from sklearn import svm
import joblib


import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.features import geometry_mask




# model specific info
from rural_beauty.config import data_dir, models_dir, get_extracted_points_paths



from rasterio.warp import calculate_default_transform, reproject, Resampling


def subset_normalize_and_align_rasters(raster_paths:dict, output_paths:dict) -> None: 
    """
    Subset all rasters to the largest common area among inputs,
    align them to the same resolution and extent, and normalize the data.

    Args:
    - raster_paths (dict): Dictionary of variable names and file paths to the input rasters.
    - output_paths (dict): Dictionary of variable names and output file paths to save the processed rasters.

    Returns:
    - None
    """

    # Open all rasters and get their bounds
    bounds_list = []
    transforms = []
    crs = None

    for path in raster_paths.values():
        with rasterio.open(path) as src:
            bounds_list.append(src.bounds)
            transforms.append(src.transform)
            if crs is None:
                crs = src.crs  # Assuming all rasters are in the same CRS, else reproject would be needed.

    # Calculate the largest common intersection (extent)
    intersection_bounds = (
        max([b.left for b in bounds_list]),   # Left bound
        max([b.bottom for b in bounds_list]), # Bottom bound
        min([b.right for b in bounds_list]),  # Right bound
        min([b.top for b in bounds_list])     # Top bound
    )

    ref_resolution = 1000
    pixel_width = ref_resolution
    pixel_height = ref_resolution
    width = int((intersection_bounds[2] - intersection_bounds[0]) / pixel_width)
    height = int((intersection_bounds[3] - intersection_bounds[1]) / pixel_height)

    # Align rasters to this common extent
    for (varname, raster_path), output_path in zip(raster_paths.items(), output_paths.values()):
        if os.path.exists(output_path):
            print(f"Skipping {output_path}, file already exists.")
            continue

        # make output dir if it doesn't exist already. 
        os.makedirs(output_path.parent, exist_ok=True)

        with rasterio.open(raster_path) as src:
            # Calculate the transform for the common extent
            transform, _, _ = calculate_default_transform(
                src.crs, src.crs, width=width, height=height, 
                left=intersection_bounds[0], bottom=intersection_bounds[1], 
                right=intersection_bounds[2], top=intersection_bounds[3])

            # Create a new dataset with the common extent
            kwargs = src.meta.copy()
            kwargs.update({
                'height': height,
                'width': width,
                'transform': transform
            })
            print(f"Aligning and normalizing {raster_path}. Dimensions: {width}, {height}")

            # Read, reproject, and normalize the data, then save
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):  # Reproject and normalize all bands
                    # Allocate an array to hold the reprojected data
                    reprojected_data = np.empty((height, width), dtype=src.dtypes[i - 1])
                    
                    reproject(
                        source=rasterio.band(src, i),
                        destination=reprojected_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest  # You can change this to another method if needed
                    )

                    # Write the normalized data to the output file
                    dst.write(reprojected_data, i)


# Function to read and check for values outside the valid range in predictor GeoTIFFs
def check_invalid_values(predictor_paths):
    for var_name, path in predictor_paths.items():
        with rasterio.open(path) as tif:
            data = tif.read(1)  # Read the first (and only) band
            if np.any(np.isnan(data)):
                print(f"Warning: {var_name} contains NaN values.")
            if np.any(np.isinf(data)):
                print(f"Warning: {var_name} contains infinity values.")
            if np.any(data > np.finfo(np.float32).max):
                print(f"Warning: {var_name} contains values too large for dtype 'float32'.")
            if np.any(data < np.finfo(np.float32).min):
                print(f"Warning: {var_name} contains values too small for dtype 'float32'.")


# Function to create a prediction GeoTIFF within a specified polygon
def create_prediction_geotiff(model, predictor_paths, output_raster_path, polygon_gdf):
    """
    Generate a prediction GeoTIFF using a trained RandomForest model and a set of predictor rasters,
    but only predict values within a given polygon.

    Args:
    - model: A fitted RandomForest model.
    - predictor_paths: Dictionary of predictor variable names and their corresponding raster paths.
    - output_raster_path: File path to save the output prediction raster.
    - polygon_gdf: A GeoDataFrame containing a single polygon.

    """
    # Read the first GeoTIFF to get metadata

    nodata_value = -99

    with rasterio.open(list(predictor_paths.values())[0]) as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            count=1,  # single band for prediction
            nodata=nodata_value  # Set the nodata value for the output
        )

        # Convert the polygon to the same CRS as the raster
        polygon = polygon_gdf.to_crs(src.crs)

        # Create a mask from the polygon to define the area for prediction
        mask = geometry_mask([geometry for geometry in polygon.geometry],
                            transform=src.transform, invert=True,
                            out_shape=(src.height, src.width))

        # Read all predictor GeoTIFFs and stack them into a 3D array
        predictors = []
        combined_mask = mask  # Initialize combined mask with polygon mask
        for path in predictor_paths.values():
            with rasterio.open(path) as tif:
                data = tif.read(1)
                data = np.where(data == tif.nodata, np.nan, data)
                predictors.append(data)
                # Update combined mask to consider areas with NA values
                combined_mask &= ~np.isnan(data)  # True where data is valid

        # Convert list of 2D arrays into a single 3D array (stacked along axis 0)
        predictors = np.stack(predictors, axis=2)
        n_rows, n_cols, n_features = predictors.shape

        # Mask the predictors to ignore areas outside the polygon or where values are NA
        predictors[~combined_mask] = np.nan

        # Flatten the masked 3D array for prediction, ignoring NaN values
        valid_data = predictors[combined_mask].reshape(-1, n_features)
        predictions = model.predict(valid_data)

        # Create an empty prediction array and fill it only in masked areas
        prediction_array = np.full((n_rows, n_cols), nodata_value, dtype=np.int32)  # Use -9999 for nodata
        prediction_array[combined_mask] = predictions.astype(np.int32)

    # Write the prediction array to a new GeoTIFF
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(prediction_array, 1)

    print(f"Finished writing the prediction to {output_raster_path}")


def parse_folder_name(model_basename):  
    parts = model_basename.split("__")
    country, target_variable, sampling_method, model_class, class_balance, sugar = parts
    number_classes = int(sugar[0])
    return country, target_variable, sampling_method, model_class, class_balance, sugar, number_classes


def print_tif_shapes(raster_paths):
    sizes = []
    for varname, path in raster_paths.items():
        with rasterio.open(path) as src:
           
            width, height = src.width, src.height
            bounds = src.bounds
            size = width * height
            sizes.append(size)
            print(f"Raster: {varname}")
            print(f"  - Shape (Width x Height): {width} x {height}")
            print(f"  - Extent (Bounds): {bounds}\n")
    
    print(min(sizes))

def main(model_folder):
    model_folder = Path(model_folder)
    model_basename = os.path.basename(model_folder)
   
    country, target_variable, sampling_method, model_class, class_balance, sugar, number_classes = parse_folder_name(model_basename)

    # print(f"Country: {country}")
    # print(f"Target Variable: {target_variable}")
    # print(f"Sampling Method: {sampling_method}")
    # print(f"Model Class: {model_class}")
    # print(f"Class Balance: {class_balance}")
    # print(f"Number of Classes: {number_classes}")
    # print(f"Sugar: {sugar}")

    # Check if teh Model folder exists. 
    if not os.path.exists(models_dir / model_basename):
        raise ValueError(f"Invalid argument: The model directory '{models_dir / model_basename}' does not exist. Have you run this exact model?")
    
    # get the raster data we trained with.
    _, _, coords_path, feature_paths = get_extracted_points_paths(country, target_variable, sampling_method)


    model_path = model_folder / 'model.pkl'

    output_raster_path = model_folder / 'prediction.tif'

    # TODO change to from config import NUTS_EU
    boundary_gpd_path = data_dir / 'cleaned' / 'NUTS' / 'EU_main.geojson'
    # load boundary polygon
    boundary_gpd  = gpd.read_file(boundary_gpd_path)


    # load the model, the scaling and the significant coefficients. 
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the model is trained and the file exists.")


    # Load the features paths dict
    try:
        with open(feature_paths, 'r') as input_file:
            feature_filepaths = json.load(input_file)
            feature_filepaths = {k: Path(v) for k, v in feature_filepaths.items()}
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature paths file not found at {feature_paths}.")

    significant_filepaths = feature_filepaths
    del significant_filepaths['hemero_1']

    # New base directory for the adjusted rasters
    new_base_dir =  data_dir / 'forprediction' / 'EU'

    new_base_dir.mkdir(parents=True, exist_ok=True)

    # Create new list of file paths for the adjusted rasters
    adjusted_raster_paths = {var:new_base_dir / f"{raster_path.name}" for var, raster_path in significant_filepaths.items()}
    
    # print_tif_shapes(significant_filepaths)
    
    subset_normalize_and_align_rasters(significant_filepaths, adjusted_raster_paths)
    
    check_invalid_values({'dem_1': significant_filepaths['dem_1']})
    
    create_prediction_geotiff(model, adjusted_raster_paths, output_raster_path, boundary_gpd)
    

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Script to predict EU from a model folder")
    
    # Add arguments
    parser.add_argument(
        "model_folder",
        type=str,
        help="example: data/models/DE__unique__random_pixels__XGB__asis__7_271124"
    )

    
    # Parse arguments
    args = parser.parse_args()
    
    # Pass arguments to main function
    main(args.model_folder)

    