#config.py

from pathlib import Path
import os

package_dir = Path(os.path.dirname(os.path.dirname(__file__)))
data_dir = package_dir.parent / 'data'



# input data
## Landschaftsbild / Ratings from the German Project
bild_vector_dir = data_dir / 'raw' / 'landschaftsbild'
bild_raster_dir = data_dir / 'cleaned' / 'landschaftsbild'
beauty_raster = data_dir / 'cleaned' / 'landschaftsbild' / 'schoenheit_DE.tif'

## Corine Land Cover
CLC_EU = data_dir / 'raw' / 'clc' / 'U2018_CLC2018_V2020_20u1.tif'
CLC_DE = data_dir / 'raw' / 'clc' / 'U2018_DE.tif'

### Boolean layers folder
CLC_boolean_layers_DE_dir = data_dir / 'processed' / 'clc' / 'single_layers'

### Layer coverage 1x1km
CLC_coverage_DE_dir = data_dir / 'cleaned' / 'clc' / 'layer_coverage_DE'

## Digital Elevation Model
### Raw Elevation Data
DEM_EU = data_dir / 'raw' / 'dem'/ 'EU_DEM_EU.tif'
### 1x1km Germany max - min elevation
DEM_DE_range = data_dir / 'cleaned' / 'dem' / 'DEM_DE_range.tif'

##OSM
### Geofabrik Germany
OSM_full_DE = data_dir / 'raw' / 'osm' / 'germany-latest.osm.pbf'
### Geofabrik EU
OSM_full_EU = data_dir / 'raw' / 'osm' / 'europe-latest.osm.pbf'

### Powerlines
powerlines_DE_vector = data_dir / 'processed' / 'osm' / 'powerlines_DE_4647.geojson'
powerlines_DE_raster = data_dir / 'cleaned' / 'osm' / 'len_powerlines_DE_4647.tif'
# powerlines_EU_vector = data_dir / 'processed' / 'osm' / 'powerlines_EU_4647.geojson'
# powerlines_EU_raster = data_dir / 'cleaned' / 'osm' / 'powerlines_EU_4647.tif'

### streets
streets_DE_vector = data_dir / 'processed' / 'osm' / 'streets_DE_4647.geojson'
streets_DE_raster = data_dir / 'cleaned' / 'osm' / 'len_streets_DE_4647.tif'

### windpower
windpower_DE_vector = data_dir / 'processed' / 'osm' / 'windpowerplants_DE_4647.geojson'
windpower_DE_raster = data_dir / 'cleaned' / 'osm' / 'freq_windpowerplants_DE_4647.tif'


## Hemerobie Index by IOER
heme2012_DE = data_dir / 'raw' / 'hemerobie_IOER' / 'heme2012.tif'
heme2012_DE_repojected = data_dir / 'cleaned' / 'hemerobie_IOER' / 'heme2012.tif'
