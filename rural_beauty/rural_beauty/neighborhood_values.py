import numpy as np
import rasterio
import os
from pathlib import Path


from rural_beauty.config import CLC_coverage_EU_dir, DEM_EU_range_scaled, windpower_EU_raster_scaled, streets_EU_raster_scaled, protected_raster_scaled
from rural_beauty import blur_circular_disc


def main(CLC_coverage_EU_dir):
    layers = [item for item in  os.listdir(CLC_coverage_EU_dir) if item.endswith('.tif')]
    print(f"For the Neighborhood we consider {len(layers)} layers in {CLC_coverage_EU_dir}")

    zone_dict = define_zone_dict(layers)

    write_specific_zone_mean(zone_dict)

    print(f"Neighborhood rasters created")




def define_zone_dict(layers) -> dict:

    # these are the hand defined zones from the German model
    zone_dict = {
    DEM_EU_range_scaled                    : ["zone2", "zone3","zone1_2", "zone3_4"], 
    CLC_coverage_EU_dir / "code_obst.tif"  : ["zone1_4"], 
    CLC_coverage_EU_dir / "code_wald.tif"  : ["zone1_4"],
    CLC_coverage_EU_dir / "code_natgru.tif": ["zone2", "zone1_2"], 
    CLC_coverage_EU_dir / "code_acker.tif" : ["zone1_4", "zone1_2"], 
    CLC_coverage_EU_dir / "code_stoer.tif" : ["zone1_2", "zone2", "zone3"], 
    CLC_coverage_EU_dir / "code_noveg.tif" : ["zone2"], 
    windpower_EU_raster_scaled             : ["zone1_4"], 
    streets_EU_raster_scaled               : ["zone1_2"], 
    protected_raster_scaled                : ["zone2", "zone3"]
    }

    # here we use the kitchin sink appraoch and add every zone (but not zone combinations like 1_4)
    added_zones = ["zone2", "zone3"]
    
    for layer in layers:
        entry = zone_dict.get(CLC_coverage_EU_dir / layer)
        if entry is None:
            zone_dict[CLC_coverage_EU_dir / layer] =  added_zones
        else:
            zone_dict[CLC_coverage_EU_dir / layer] = list(set(zone_dict[CLC_coverage_EU_dir / layer] + added_zones))

    return zone_dict


def write_specific_zone_mean(input_dict:str) -> None:
    overall_radius = 10
    zone_kernels = {
    "zone4" : blur_circular_disc.create_circular_disc(9.9, overall_radius) - blur_circular_disc.create_circular_disc(4.9, overall_radius),
    "zone3" : blur_circular_disc.create_circular_disc(4.9, 5) - blur_circular_disc.create_circular_disc(1.9, 5),
    "zone2" : blur_circular_disc.create_circular_disc(1.9, 2) - blur_circular_disc.create_circular_disc(0.9, 2),
    "zone1_4" : blur_circular_disc.create_circular_disc(9.9, overall_radius),
    "zone1_2" : blur_circular_disc.create_circular_disc(1.9, 2),
    "zone3_4" : blur_circular_disc.create_circular_disc(9.9, overall_radius) - blur_circular_disc.create_circular_disc(1.9, overall_radius)
    }
    
    for input_path, zone_names in input_dict.items():
        
        with rasterio.open(input_path) as raster_in:
            for zone_name in zone_names:
                # Define the output path within the new directory
                output_path = modify_filename(input_path, zone_name, 'neighborhood')

                if zone_name == 'zone1':
                    print(f"Zone 1 is the same as input data. Skipped :{output_path}")
                    continue
                
                if os.path.exists(output_path):
                    print(f"{output_path} already exists skipping...")
                    continue

                kernel = zone_kernels[zone_name]
                meta = raster_in.meta.copy()
                na_value = raster_in.nodata
                array_in = raster_in.read(1)  # Reading the first band
    


    
                print(output_path)
                output_array = blur_circular_disc.get_zone_mean(array_in, na_value, kernel)
        
                # Create a new raster file for output
                meta.update(dtype=rasterio.float64, count=1)  # Update meta if necessary
        
                with rasterio.open(output_path, 'w', **meta) as raster_out:
                    raster_out.write(output_array, 1)  # Write output array to the first band



def modify_filename(file_path:str, add_string:str, new_folder:str) -> str:
    # Split the filename and the extension
    base, extension = os.path.splitext(file_path)

    # Get directory and basename
    dir_name, base_name = os.path.split(base)

    # Add the string to the basename
    new_base_name = f"{base_name}_{add_string}"

    # Construct the new path
    new_full_path = os.path.join(dir_name, new_folder, new_base_name + extension)

    # Ensure the directory exists
    new_directory = os.path.dirname(new_full_path)
    os.makedirs(new_directory, exist_ok=True)

    return new_full_path

if __name__ == "__main__":
    main()