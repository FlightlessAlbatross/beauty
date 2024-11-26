import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
from pathlib import Path
import json
import rasterio
from rasterio.mask import mask

# Configuration imports
from rural_beauty.config import (
    unique_raster,
    beauty_raster,
    diverse_raster,
    NUTS_DE,
    CLC_coverage_EU_dir,
    BFN_features_diverse,
    BFN_features_unique,
    BFN_features_beauty,
    predictors_DE,
    outcome_DE,
    coords_DE,
    feature_paths_DE,
)

NUTS_path = NUTS_DE

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

# Verify all files exist
some_missing = False
for label, path in features.items():
    if not os.path.exists(path):
        some_missing = True
        print(f"{path} is missing")

if not some_missing:
    print("All files exist")

# Read the NUTS geometry
NUTS = gpd.read_file(NUTS_path)
polygon = NUTS.geometry.iloc[0]

# Function to extract all pixel values within a polygon from a raster
def extract_raster_values_within_polygon(raster_path, polygon):
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

        # Create a DataFrame
        coords_df = pd.DataFrame({
            'x_coord': coords[0],
            'y_coord': coords[1],
        })
        values_df = pd.DataFrame({'value': values})

    return coords_df, values_df

# Extract the outcome raster values
coords_df, outcome_values = extract_raster_values_within_polygon(beauty_raster, polygon)
outcome_df = outcome_values.rename(columns={'value': 'beauty'})

# Process all predictors
predictor_dataframes = []
for label, path in features.items():
    _, predictor_values = extract_raster_values_within_polygon(path, polygon)
    predictor_values.rename(columns={'value': label}, inplace=True)
    predictor_dataframes.append(predictor_values)

# Combine predictors into a single DataFrame
predictors_df = pd.concat(predictor_dataframes, axis=1)

# Ensure the directory exists
os.makedirs(os.path.dirname(predictors_DE), exist_ok=True)

# Write DataFrames to CSV files
coords_df.to_csv(coords_DE, index=False, float_format='%.15g')
outcome_df.to_csv(outcome_DE, index=False, float_format='%.15g')
predictors_df.to_csv(predictors_DE, index=False, float_format='%.15g')

# Save feature paths to a JSON file
features_out = {k: str(v) for k, v in features.items()}
with open(feature_paths_DE, 'w') as output:
    json.dump(features_out, output, indent=4)

print(f"Coordinates saved to: {coords_DE}")
print(f"Outcome saved to: {outcome_DE}")
print(f"Predictors saved to: {predictors_DE}")
