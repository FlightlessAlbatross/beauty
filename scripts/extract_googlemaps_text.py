import json
import pandas as pd
import pyproj
from shapely.geometry import Point, mapping
from pathlib import Path

def text_to_geojson(input_file_path, output_file_path, crs_from="EPSG:4326", crs_to="EPSG:4647"):
    """
    Converts a text file with name, latitude, and longitude columns into a GeoJSON file
    with coordinates transformed to a target CRS.

    Parameters:
    - input_file_path: str, path to the input text file (CSV-like, with name, latitude, longitude).
    - output_file_path: str, path where the GeoJSON file will be written.
    - crs_from: str, source CRS (default is "EPSG:4326").
    - crs_to: str, target CRS (default is "EPSG:4647").

    Returns:
    - None
    """
    # Load the data from the text file
    data = pd.read_csv(input_file_path, header=None, names=["name", "latitude", "longitude"])
    
    # Initialize the transformer
    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    
    # Prepare GeoJSON features
    features = []
    for _, row in data.iterrows():
        # Transform coordinates
        x, y = transformer.transform(row["longitude"], row["latitude"])
        # Create a feature
        features.append({
            "type": "Feature",
            "geometry": mapping(Point(x, y)),
            "properties": {"name": row["name"]}
        })
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Write GeoJSON to file
    with open(output_file_path, "w") as f:
        json.dump(geojson, f, indent=4)
    print(f"GeoJSON written to {output_file_path}")

def process_directory(input_dir, output_dir, crs_from="EPSG:3857", crs_to="EPSG:4647"):
    """
    Processes all text files in a directory, converting them to GeoJSON.
    
    Parameters:
    - input_dir: str, directory containing text files.
    - output_dir: str, directory where GeoJSON files will be saved.
    - crs_from: str, source CRS (default is "EPSG:4326").
    - crs_to: str, target CRS (default is "EPSG:4647").
    
    Returns:
    - None
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for file in input_path.glob("*.txt"):
        output_file = output_path / (file.stem + ".geojson")
        text_to_geojson(file, output_file, crs_from, crs_to)

# Example usage:
if __name__ == "__main__":
     #python3 scripts/extract_googlemaps_text.py data/raw/manual_validation/nice.csv data/cleaned/manual_validation.geojson
    input_directory = "data/raw/manual_validation"
    output_directory = "data/cleaned/manual_validation"
    process_directory(input_directory, output_directory)
