#!/bin/bash

EPSG_CODE="EPSG:4647"
ogr2ogr -f "GeoJSON" -t_srs $EPSG_CODE -where "highway='trunk' OR highway='motorway' OR highway='primary' OR highway='secondary'" streets.geojson germany-latest.osm.pbf lines
ogr2ogr -f "GeoJSON" -t_srs $EPSG_CODE powerlines.geojson germany-latest.osm.pbf lines -where "other_tags LIKE '%\"power\"=>\"line\"%'"
ogr2ogr -f "GeoJSON" -t_srs $EPSG_CODE windpowerplants.geojson germany-latest.osm.pbf points -where "other_tags LIKE '%\"power\"=>\"generator\"%' AND other_tags LIKE '%\"generator:source\"=>\"wind\"%'"