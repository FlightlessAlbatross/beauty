import argparse
import rasterio
import joblib
import pandas as pd
import json
from pathlib import Path
import numpy as np
import os

from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.features import geometry_mask

import geopandas as gpd


# model specific info
from rural_beauty.config import data_dir, models_dir 
import importlib


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



def main(country, sugar):

    module_name = "rural_beauty.config"

    if country not in ["UK", "DE"]:
        raise ValueError(f"Invalid country argument: {country}. Supported values are 'UK' or 'DE'.")


    if country == 'UK':
        para_outcome = 'scenic'
    
    if country == 'DE':
        para_outcome = 'beauty'
        print('The default outcome variable for Germany (DE) is beauty as it compares best with scenic.')

    para_type = 'randomforest'
    model_basename = f"{country}_{para_outcome}_{para_type}_{sugar}"
    model_folder = models_dir / model_basename

    print(f"The model name based on the arguments is: {model_basename}")

    if not os.path.exists(models_dir / model_basename):
        raise ValueError(f"Invalid argument: The model directory '{models_dir / model_basename}' does not exist. Have you run this exact model?")


    # Import the module
    config_module = importlib.import_module(module_name)

    # Dynamically construct the variable name
    feature_paths   = getattr(config_module, f"feature_paths_{country}")

    model_path = model_folder / 'model.pkl'
    significant_coefs_path = model_folder / "significant_coefs.csv"

    output_raster_path = model_folder / 'prediction.tif'

    boundary_gpd_path = data_dir / 'cleaned' / 'NUTS' / 'EU_main.geojson'


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

    significant_coefs = pd.read_csv(significant_coefs_path)
    significant_filepaths = {feature : feature_filepaths[feature] for feature in significant_coefs.Feature}

    # New base directory for the adjusted rasters
    new_base_dir =  data_dir / 'forprediction'

    new_base_dir.mkdir(parents=True, exist_ok=True)

    # Create new list of file paths for the adjusted rasters
    adjusted_raster_paths = {var:new_base_dir / model_basename / f"{raster_path.name}" for var, raster_path in significant_filepaths.items()}

    os.makedirs(new_base_dir / model_basename, exist_ok=True)


    subset_normalize_and_align_rasters(significant_filepaths, adjusted_raster_paths)

    check_invalid_values({'dem_1': significant_filepaths['dem_1']})

    # load boundary polygon
    boundary_gpd  = gpd.read_file(boundary_gpd_path)

    create_prediction_geotiff(model, adjusted_raster_paths, output_raster_path, boundary_gpd)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Script to run model generation based on country, type, and sugar.")
    
    # Add arguments
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country code (e.g., 'UK', 'DE', etc.)."
    )
    parser.add_argument(
        "--sugar",
        type=str,
        required=True,
        help="Identifier for the model (e.g., '111124')."
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Pass arguments to main function
    main(args.country, args.sugar)