#config.py

from pathlib import Path
import os
from typing import Tuple

package_dir = Path(os.path.dirname(os.path.dirname(__file__)))
data_dir = package_dir.parent / 'data'



# input data
## Protected areas
protected0 = data_dir /  'raw'/ 'protected' / 'WDPA_Oct2024_Public_shp_0/WDPA_Oct2024_Public_shp-polygons.shp'
protected1 = data_dir /  'raw'/ 'protected' / 'WDPA_Oct2024_Public_shp_1/WDPA_Oct2024_Public_shp-polygons.shp'
protected2 = data_dir /  'raw'/ 'protected' / 'WDPA_Oct2024_Public_shp_2/WDPA_Oct2024_Public_shp-polygons.shp'

protected_EU = data_dir / 'processed' / 'protected' /  'WDPA_merged_4647.shp'
protected_raster = data_dir / 'processed' / 'protected' / 'WDPA_area.tif'
protected_raster_scaled = data_dir / 'cleaned' / 'protected' / 'WDPA_area_scaled.tif'


## Landschaftsbild / Ratings from the German Project
bild_vector_dir = data_dir / 'raw' / 'landschaftsbild'
bild_raster_dir = data_dir / 'cleaned' / 'landschaftsbild'
beauty_raster = data_dir / 'cleaned' / 'landschaftsbild' / 'schoenheit_DE.tif'
unique_raster = data_dir / 'cleaned' / 'landschaftsbild' / 'eigenart_DE.tif'
diverse_raster = data_dir / 'cleaned' / 'landschaftsbild' / 'vielfalt_DE.tif'
DE_outcome_raster_paths = {'unique': unique_raster, 
                           'diverse': diverse_raster, 
                           'beauty': beauty_raster}

## UK data
UK_scenic_raw    = data_dir / "raw"       / "scenicornot" / "votes.tsv"
UK_scenic_points = data_dir / "processed" / "scenicornot" / "scenic.geojson"
UK_scenic_raster = data_dir / "processed"   / "scenicornot" / "scenic.tif"

def get_target_variable_raster_path(country, target_variable = None) -> Path:
    
    if country == 'DE':
      if target_variable in DE_outcome_raster_paths:        
         return DE_outcome_raster_paths[target_variable]
      else:
         raise ValueError('The target variable {target_variable} is not valid for DE')
    if country == 'UK':
       return UK_scenic_raster
        


## Corine Land Cover
CLC_EU = data_dir / 'raw' / 'clc' / 'U2018_CLC2018_V2020_20u1.tif'
CLC_DE = data_dir / 'raw' / 'clc' / 'U2018_EU.tif'

### Boolean layers folder
CLC_boolean_layers_dir = data_dir / 'processed' / 'clc' / 'single_layers'

### Layer coverage 1x1km
CLC_coverage_DE_dir = data_dir / 'cleaned' / 'clc' / 'layer_coverage_DE'
CLC_coverage_EU_dir = data_dir / 'cleaned' / 'clc' / 'layer_coverage_EU'

## Digital Elevation Model
### Raw Elevation Data
DEM_EU = data_dir / 'raw' / 'dem'/ 'EU_DEM_EU.tif'
### 1x1km Germany max - min elevation
DEM_DE_range = data_dir / 'processed' / 'dem' / 'DEM_DE_range.tif'
DEM_EU_range = data_dir / 'processed' / 'dem' / 'DEM_EU_range.tif'
DEM_EU_range_scaled = data_dir / 'cleaned' / 'dem' / 'DEM_EU_range_scaled.tif'
DEM_DE_range_scaled = data_dir / 'cleaned' / 'dem' / 'DEM_DE_range_scaled.tif'

##OSM
### Geofabrik Germany
OSM_full_DE = data_dir / 'raw' / 'osm' / 'germany-latest.osm.pbf'
### Geofabrik EU
OSM_full_EU = data_dir / 'raw' / 'osm' / 'europe-latest.osm.pbf'

### Powerlines
powerlines_DE_vector = data_dir / 'processed' / 'osm' / 'powerlines_DE_4647.geojson'
powerlines_DE_raster = data_dir / 'processed' / 'osm' / 'len_powerlines_DE_4647.tif'

powerlines_EU_vector = data_dir / 'processed' / 'osm' / 'powerlines_EU_4647.geojson'
powerlines_EU_raster = data_dir / 'processed' / 'osm' / 'len_powerlines_EU_4647.tif'
powerlines_EU_raster_scaled = data_dir / 'cleaned' / 'osm' / 'len_scaled_powerlines_EU_4647.tif'


### streets
streets_DE_vector = data_dir / 'processed' / 'osm' / 'streets_DE_4647.geojson'
streets_DE_raster = data_dir / 'processed' / 'osm' / 'len_streets_DE_4647.tif'

streets_EU_vector = data_dir / 'processed' / 'osm' / 'streets_EU_4647.geojson'
streets_EU_raster = data_dir / 'processed' / 'osm' / 'len_streets_EU_4647.tif'
streets_EU_raster_scaled = data_dir / 'cleaned' / 'osm' / 'len_scaled_streets_EU_4647.tif'

### windpower
windpower_DE_vector = data_dir / 'processed' / 'osm' / 'windpowerplants_DE_4647.geojson'
windpower_DE_raster = data_dir / 'processed' / 'osm' / 'freq_windpowerplants_DE_4647.tif'

windpower_EU_vector = data_dir / 'processed' / 'osm' / 'windpowerplants_EU_4647.geojson'
windpower_EU_raster = data_dir / 'processed' / 'osm' / 'freq_windpowerplants_EU_4647.tif'
windpower_EU_raster_scaled = data_dir / 'cleaned' / 'osm' / 'freq_scaled_windpowerplants_EU_4647.tif'


## Hemerobie Index by IOER
heme2012_DE = data_dir / 'raw' / 'hemerobie_IOER' / 'heme2012.tif'
heme2012_DE_repojected = data_dir / 'cleaned' / 'hemerobie_IOER' / 'heme2012.tif'



#auxiliary data
## NUTS
NUTS_DE = data_dir / 'cleaned' / 'NUTS' / 'DE.geojson'
NUTS_EU = data_dir / 'cleaned' / 'NUTS' / 'EU.geojson'
NUTS_UK = data_dir / 'cleaned' / 'NUTS' / 'UK_without_NI.geojson'

# modeling data

# here we will save random points, model pkl mean and sd of how we scaled the input data, so we can replicate it for predictions
models_dir  = data_dir / 'models'
extracted_points_dir = models_dir / '__extracted_points'

def get_extracted_points_paths(country, target_variable, sampling_method) -> Tuple[Path, Path, Path, Path]:
    """
    Constructs and returns the paths for predictors, outcome, coordinates, and feature paths 
    for a given country and sampling method.

    Parameters:
    ----------
    country : str
        The country code (e.g., 'DE', 'UK', 'DEUK').
    sampling_method : str
        The sampling method used (e.g., 'random_pixels', 'all_pixels', 'pooled_pixels_random_sample', pooled_pixels_all_points, pooled_pixels_all_pixels,  'individual_images_full_sample').
        Not every sampling method is available for every country

    Returns:
    -------
    Tuple[Path, Path, Path, Path]
        A tuple containing the following paths:
        - predictors_path: Path to the predictors file.
        - outcome_path: Path to the outcome file.
        - coords_path: Path to the coordinates file.
        - feature_paths_path: Path to the feature paths JSON file.
    """
    base_path = extracted_points_dir / country / target_variable / sampling_method
    return (base_path / 'predictors.csv', 
            base_path / 'outcome.csv',
            base_path / 'coords.csv',
            base_path / 'feature_paths.json')


# predictors_DE    = extracted_points_dir / 'DE' / 'predictors.csv'
# outcome_DE       = extracted_points_dir / 'DE' / 'outcome.csv'
# coords_DE        = extracted_points_dir / 'DE' / 'coords.csv'
# feature_paths_DE = extracted_points_dir / 'DE' / 'feature_paths.json'

# predictors_UK    = extracted_points_dir / 'UK' / 'predictors.csv'
# outcome_UK       = extracted_points_dir / 'UK' / 'outcome.csv'
# coords_UK        = extracted_points_dir / 'UK' / 'coords.csv'
# feature_paths_UK = extracted_points_dir / 'UK' / 'feature_paths.json'




# included features from the BFN (Bundesamt fuer Naturschutz) Beauty source (data_dir / raw/landschaftsbild/BfN*.pdf)

BFN_features_beauty = {
 'dem_1_2'  : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_scaled_zone1_2.tif',
 'dem_3_4'  : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_scaled_zone3_4.tif', 
 'obst_1_4' : CLC_coverage_EU_dir / 'neighborhood/code_obst_zone1_4.tif',
 'wald_1_4' : CLC_coverage_EU_dir / 'neighborhood/code_wald_zone1_4.tif',
 'natgru_2' : CLC_coverage_EU_dir / 'neighborhood/code_natgru_zone2.tif',
 'acker_1_4': CLC_coverage_EU_dir / 'neighborhood/code_acker_zone1_4.tif',
 'stoer_1'  : CLC_coverage_EU_dir / 'code_stoer.tif',
 'stoer_2'  : CLC_coverage_EU_dir / 'neighborhood/code_stoer_zone2.tif',
 'stoer_3'  : CLC_coverage_EU_dir / 'neighborhood/code_stoer_zone3.tif',
 'noveg_2'  : CLC_coverage_EU_dir / 'neighborhood/code_noveg_zone2.tif', 
 'seemee_1' : CLC_coverage_EU_dir / 'code_seemee.tif',
 'spfr_1'   : CLC_coverage_EU_dir / 'code_spfr.tif',
 'heide_1'  : CLC_coverage_EU_dir / 'code_heide.tif',
 'weanl_1_4': data_dir / 'cleaned/osm/neighborhood/freq_scaled_windpowerplants_EU_4647_zone1_4.tif', 
 'stra_1_2' : data_dir / 'cleaned/osm/neighborhood/len_scaled_streets_EU_4647_zone1_2.tif', 
 'leit_1'   : powerlines_EU_raster_scaled, 
 'hemero_1' : heme2012_DE_repojected
           }

BFN_features_scenic = BFN_features_beauty

BFN_features_unique = {
 'dem_1'      : DEM_EU_range_scaled,
 'dem_3'      : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_scaled_zone3.tif',
 'seemee_1'   : CLC_coverage_EU_dir / 'code_seemee.tif',
 'heide_1'    : CLC_coverage_EU_dir / 'code_heide.tif',
 'sgall_1'    : data_dir / protected_raster_scaled,
 'natgru_1_2' : CLC_coverage_EU_dir / 'neighborhood/code_natgru_zone1_2.tif',
 'wein_1'     : CLC_coverage_EU_dir / 'code_wein.tif', 
 'acker_1_2'  : CLC_coverage_EU_dir / 'neighborhood/code_acker_zone1_2.tif',
 'stoer_1_2'  : CLC_coverage_EU_dir / 'neighborhood/code_stoer_zone1_2.tif',
 'stra_1'     : data_dir / 'cleaned/osm/len_scaled_streets_EU_4647.tif', 
 'leit_1'     : powerlines_EU_raster_scaled
           }


# incomplete: missing Diversity scores. 
BFN_features_diverse = {
 'dem_1'    : DEM_EU_range_scaled,
 'dem_2'    : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_scaled_zone2.tif',
 'dem_3'    : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_scaled_zone3.tif',
 'seemee_1' : CLC_coverage_EU_dir / 'code_seemee.tif',
 'wald_1'   : CLC_coverage_EU_dir / 'code_wald.tif',
 'natgru_2' : CLC_coverage_EU_dir / 'neighborhood/code_natgru_zone_2.tif',
 'obst_1_4' : CLC_coverage_EU_dir / 'neighborhood/code_obst_zone_1_4.tif',
 'stoer_1_2': CLC_coverage_EU_dir / 'neighborhood/code_stoer_zone1_2.tif',
 'hemero_1' : heme2012_DE_repojected, 
 'stra_1_2' : data_dir / 'cleaned/osm/neighborhood/len_scaled_streets_EU_4647_zone_1_2.tif', 
 'acker_1_4': CLC_coverage_EU_dir / 'neighborhood/code_acker_zone1_4.tif',
 'noveg_2'  : CLC_coverage_EU_dir / 'neighborhood/code_noveg_zone_2.tif', 
 'leit_1'   : powerlines_EU_raster_scaled
 }





# these are the beta coefs
BFN_coefs_unique = {
  'dem_1'      : 0.164,
  'dem_3'      : 0.392,
  'seemee_1'   : 0.202,
  'heide_1'    : 0.166,
  'sgall_1'    : 0.108,
  'natgru_1_2' : 0.080,
  'wein_1'     : 0.054,
  'acker_1_2'  :-0.199,
  'stoer_1_2'  :-0.184,
  'stra_1'     :-0.142,
  'leit_1'     :-0.070,
}
BFN_constant_unique = 0
