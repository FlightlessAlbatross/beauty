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
DEM_DE_range = data_dir / 'cleaned' / 'dem' / 'DEM_DE_range.tif'
DEM_EU_range = data_dir / 'cleaned' / 'dem' / 'DEM_EU_range.tif'

##OSM
### Geofabrik Germany
OSM_full_DE = data_dir / 'raw' / 'osm' / 'germany-latest.osm.pbf'
### Geofabrik EU
OSM_full_EU = data_dir / 'raw' / 'osm' / 'europe-latest.osm.pbf'

### Powerlines
powerlines_DE_vector = data_dir / 'processed' / 'osm' / 'powerlines_DE_4647.geojson'
powerlines_DE_raster = data_dir / 'cleaned' / 'osm' / 'len_powerlines_DE_4647.tif'
powerlines_EU_vector = data_dir / 'processed' / 'osm' / 'powerlines_EU_4647.geojson'
powerlines_EU_raster = data_dir / 'cleaned' / 'osm' / 'len_powerlines_EU_4647.tif'

### streets
streets_DE_vector = data_dir / 'processed' / 'osm' / 'streets_DE_4647.geojson'
streets_DE_raster = data_dir / 'cleaned' / 'osm' / 'len_streets_DE_4647.tif'
streets_EU_vector = data_dir / 'processed' / 'osm' / 'streets_EU_4647.geojson'
streets_EU_raster = data_dir / 'cleaned' / 'osm' / 'len_streets_EU_4647.tif'

### windpower
windpower_DE_vector = data_dir / 'processed' / 'osm' / 'windpowerplants_DE_4647.geojson'
windpower_DE_raster = data_dir / 'cleaned' / 'osm' / 'freq_windpowerplants_DE_4647.tif'
windpower_EU_vector = data_dir / 'processed' / 'osm' / 'windpowerplants_EU_4647.geojson'
windpower_EU_raster = data_dir / 'cleaned' / 'osm' / 'freq_windpowerplants_EU_4647.tif'

## Hemerobie Index by IOER
heme2012_DE = data_dir / 'raw' / 'hemerobie_IOER' / 'heme2012.tif'
heme2012_DE_repojected = data_dir / 'cleaned' / 'hemerobie_IOER' / 'heme2012.tif'



#auxiliary data
## NUTS
NUTS_DE = data_dir / 'cleaned' / 'NUTS' / 'DE.geojson'
NUTS_EU = data_dir / 'cleaned' / 'NUTS' / 'EU.geojson'

# modeling data
predictors_DE = data_dir / 'cleaned' / 'modeling' / 'rnd_points' / 'DE' / 'predictors.csv'
outcome_DE     = data_dir / 'cleaned' / 'modeling' / 'rnd_points' / 'DE' / 'outcome.csv'
# beauty_DE     = data_dir / 'cleaned' / 'modeling' / 'rnd_points' / 'DE' / 'beauty.csv'
# unique_DE     = data_dir / 'cleaned' / 'modeling' / 'rnd_points' / 'DE' / 'unique.csv'
# diverse_DE    = data_dir / 'cleaned' / 'modeling' / 'rnd_points' / 'DE' / 'diverse.csv'

coords_DE     = data_dir / 'cleaned' / 'modeling' / 'rnd_points' / 'DE' / 'coords.csv'
