import rasterio
import os
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm # for progress bars. 
import psutil # to check available free cores. 


def process_layer(label, values, data, meta, output_folder):
    """
    Process a single layer: create a binary mask and save it as a separate file.
    """
    mask = np.zeros_like(data, dtype=bool)
    output_path = os.path.join(output_folder, f"{label}.tif")
    if os.path.exists(output_path):
        return

    # Create mask
    if isinstance(values, list):
        for value in values:
            mask |= (data == value)
    elif isinstance(values, int):
        mask |= (data == values)
    else:
        print(f"{label} has faulty values, should be list or int")
        return

    output_data = mask.astype(np.uint8)

    # Write output raster
    with rasterio.open(output_path, 'w', **meta) as out_rst:
        out_rst.write(output_data, 1)


def parallel_process(clc_path, layers, output_folder, max_workers=4):
    with rasterio.open(clc_path) as src:
        meta = src.meta.copy()
        meta['dtype'] = 'uint8'
        data = src.read(1)

        # Prepare arguments for parallel processing
        tasks = [
            (label, values, data, meta, output_folder)
            for label, values in layers.items()
        ]

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(lambda args: process_layer(*args), tasks), 
                      total=len(tasks), desc=f"Extract a CLC Class to a boolean layer with {max_workers} threads. Existing Files are skipped."))


def main(clc_path, output_folder):
    layers = {
        "code_stadt": 1,
        "code_dorf": 2,
        "code_indgew": 3,
        "code_strbah": 4,
        "code_hafen": 5,
        "code_flug": 6,
        "code_stgrue": 10,
        "code_spfr": 11,
        "code_acker": 12,
        "code_wein": 15,
        "code_obst": 16,
        "code_weide": 18,
        "code_lwald": 23,
        "code_nwald": 24,
        "code_mwald": 25,
        "code_natgru": 26,
        "code_heide": 27,
        "code_strauc": 29,
        "code_sand": 30,
        "code_fels": 31,
        "code_noveg": 32,
        "code_suempf": 35,
        "code_moor": 36,
        "code_salzw": 37,
        "code_gezei": 39,
        "code_gwlf": 40,
        "code_seen": 41,
        "code_siedl": [1, 2],
        "code_bebau": [1, 2, 3],
        "code_abbau": [7, 8, 9],
        "code_stoer": [3, 4, 5, 6, 7, 8, 9],
        "code_obswei": [15, 16],
        "code_landwi": [12, 13, 14, 15, 16, 18],
        "code_wald": [23, 24, 25],
        "code_kraut": [26, 27, 29],
        "code_wanat": [23, 24, 25, 26, 27, 29, 30, 31, 32, 35, 36, 37, 39, 42, 43, 44],
        "code_natur": [26, 27, 29, 30, 31, 32, 35, 36, 39, 42, 43],
        "code_offen": [30, 31, 32],
        "code_selten": [27, 30, 31, 32, 35, 36, 37],
        "code_feucht": [35, 36, 37, 39],
        "code_meer": [42, 43, 44],
        "code_gewae": [40, 41, 42, 43, 44],
        "code_gewage": [39, 40, 41, 42, 43, 44],
        "code_seemee": [41, 42, 43, 44],
        "code_semage": [39, 41, 42, 43, 44],
        "code_schatt": [1, 2, 3, 15, 16, 23, 24, 25, 29],
        "code_geholz": [15, 16, 23, 24, 25, 29]
    }

    os.makedirs(output_folder, exist_ok=True)


    total_cores = psutil.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)
    available_cores = max(1, int(total_cores * (1 - cpu_usage / 100.0)))
    max_workers = min(available_cores, 8)

    parallel_process(clc_path, layers, output_folder, max_workers=max_workers)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_raster_path> <output_folder>")
        sys.exit(1)
    
    clc_path = sys.argv[1]
    output_folder = sys.argv[2]

    main(clc_path, output_folder)
    print('boolean layers were created')
