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
CLC_EU = data_dir / 'raw' / 'CLC' / 'U2018_CLC2018_V2020_20u1.tif'
CLC_DE = data_dir / 'raw' / 'CLC' / 'U2018_DE.tif'

# Digital Elevation Model
DEM_EU = data_dir / 'raw' / 'dem'/ 'EU_DEM_EU.tif'

DEM_DE_range = data_dir / 'cleaned' / 'DEM' / 'DEM_DE_range.tif'



