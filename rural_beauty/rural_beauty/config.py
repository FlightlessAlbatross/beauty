#config.py

from pathlib import Path
import os

package_dir = Path(os.path.dirname(os.path.dirname(__file__)))
data_dir = package_dir.parent / 'data'



# input data
## Protected areas
protected0 = data_dir /  'raw'/ 'protected' / 'WDPA_Oct2024_Public_shp_0/WDPA_Oct2024_Public_shp-polygons.shp'
protected1 = data_dir /  'raw'/ 'protected' / 'WDPA_Oct2024_Public_shp_1/WDPA_Oct2024_Public_shp-polygons.shp'
protected2 = data_dir /  'raw'/ 'protected' / 'WDPA_Oct2024_Public_shp_2/WDPA_Oct2024_Public_shp-polygons.shp'

protected_EU = data_dir / 'processed' / 'protected' /  'WDPA_merged_4647.shp'
protected_raster = data_dir / 'cleaned' / 'protected' / 'WDPA_area.tif'

## Landschaftsbild / Ratings from the German Project
bild_vector_dir = data_dir / 'raw' / 'landschaftsbild'
bild_raster_dir = data_dir / 'cleaned' / 'landschaftsbild'
beauty_raster = data_dir / 'cleaned' / 'landschaftsbild' / 'schoenheit_DE.tif'
unique_raster = data_dir / 'cleaned' / 'landschaftsbild' / 'eigenart_DE.tif'
diverse_raster = data_dir / 'cleaned' / 'landschaftsbild' / 'vielfalt_DE.tif'

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

# modeling data
predictors_DE = data_dir / 'models' / '_rnd_points' / 'DE' / 'predictors.csv'
outcome_DE    = data_dir  / 'models' / '_rnd_points' / 'DE' / 'outcome.csv'
coords_DE     = data_dir /  'models' / '_rnd_points' / 'DE' / 'coords.csv'
feature_paths = data_dir /  'models' / '_rnd_points' / 'DE' / 'feature_paths.json'

# here we will save random points, model pkl mean and sd of how we scaled the input data, so we can replicate it for predictions
models_dir  = data_dir / 'models'


# included features from the BFN (Bundesamt fuer Naturschutz) Beauty source (data_dir / raw/landschaftsbild/BfN*.pdf)

BFN_features_beauty = {
 'dem_1_2'  : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_zone1_2.tif',
 'dem_3_4'  : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_zone3_4.tif', 
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
 'weanl_1_4': data_dir / 'cleaned/osm/neighborhood/freq_windpowerplants_EU_4647_zone1_4.tif', 
 'stra_1_2' : data_dir / 'cleaned/osm/neighborhood/len_streets_EU_4647_zone1_2.tif', 
 'leit_1'   : powerlines_EU_raster, 
 'hemero_1' : heme2012_DE_repojected
           }


BFN_features_unique = {
 'dem_1'      : data_dir / 'cleaned/dem/DEM_EU_range.tif',
 'dem_3'      : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_zone3.tif',
 'seemee_1'   : CLC_coverage_EU_dir / 'code_seemee.tif',
 'heide_1'    : CLC_coverage_EU_dir / 'code_heide.tif',
 'sgall_1'    : data_dir / protected_raster,
 'natgru_1_2' : CLC_coverage_EU_dir / 'neighborhood/code_natgru_zone1_2.tif',
 'wein_1'     : CLC_coverage_EU_dir / 'code_wein.tif', 
 'acker_1_2'  : CLC_coverage_EU_dir / 'neighborhood/code_acker_zone1_2.tif',
 'stoer_1_2'  : CLC_coverage_EU_dir / 'neighborhood/code_stoer_zone1_2.tif',
 'stra_1'     : data_dir / 'cleaned/osm/len_streets_EU_4647.tif', 
 'leit_1'     : data_dir / 'cleaned/osm/len_powerlines_EU_4647.tif'
           }


# incomplete: missing Diversity scores. 
BFN_features_diverse = {
 'dem_1'    : data_dir / 'cleaned/dem/DEM_EU_range.tif',
 'dem_2'    : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_zone2.tif',
 'dem_3'    : data_dir / 'cleaned/dem/neighborhood/DEM_EU_range_zone3.tif',
 'seemee_1' : CLC_coverage_EU_dir / 'code_seemee.tif',
 'wald_1'   : CLC_coverage_EU_dir / 'code_wald.tif',
 'natgru_2' : CLC_coverage_EU_dir / 'neighborhood/code_natgru_zone_2.tif',
 'obst_1_4' : CLC_coverage_EU_dir / 'neighborhood/code_obst_zone_1_4.tif',
 'stoer_1_2': CLC_coverage_EU_dir / 'neighborhood/code_stoer_zone1_2.tif',
 'hemero_1' : heme2012_DE_repojected, 
 'stra_1_2' : data_dir / 'cleaned/osm/neighborhood/len_streets_EU_4647_zone_1_2.tif', 
 'acker_1_4': CLC_coverage_EU_dir / 'neighborhood/code_acker_zone1_4.tif',
 'noveg_2'  : CLC_coverage_EU_dir / 'neighborhood/code_noveg_zone_2.tif', 
 'leit_1'   : data_dir / 'cleaned/osm/neighborhood/len_powerlines_EU_4647_zone_1_2.tif'
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
