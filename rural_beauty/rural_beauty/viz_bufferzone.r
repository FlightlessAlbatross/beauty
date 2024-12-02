withbuffer_coords <- read.csv("./data/models/__extracted_points/DE/beauty/all_pixels/coords.csv")
nobuffer_coords   <- read.csv("./data/models/__extracted_points/DE/beauty/nobuffer/coords.csv")

withbuffer_features <- read.csv("./data/models/__extracted_points/DE/beauty/all_pixels/predictors.csv", na.strings = '-99')
withbuffer_outcome  <- read.csv("./data/models/__extracted_points/DE/beauty/all_pixels/outcome.csv", na.strings = '-99')

sprintf("Without a buffer the dataset has %d rows", nrow(nobuffer_coords))
sprintf("With a buffer it has %d rows", nrow(withbuffer_coords))

get_rowindes_of_bufferzone <- function(){
    !(with(withbuffer_coords, paste0(x_coord, y_coord)) %in%
      with(nobuffer_coords, paste0(x_coord, y_coord)) )
}
bufferzone_idx <- get_rowindes_of_bufferzone()

get_coordinates_in_bufferzone <- function(){
    withbuffer_coords[get_rowindes_of_bufferzone(),]}

bufferzone <- get_coordinates_in_bufferzone()


sprintf("There should be %d pixels in the bufferzone. We found %d",
 nrow(withbuffer_coords) - nrow(nobuffer_coords), nrow(bufferzone))

# check out the DEM EU values and the others too. CLC, DEM, OSM, Protected. 

features_subset_names <- c('dem_1', 'seemee_1', 'stra_1', 'sgall_1')
example_features <- withbuffer_features[bufferzone_idx, features_subset_names]

outcome_buffer <- withbuffer_outcome[bufferzone_idx,]


Map(function(col, col_name) {
    sprintf("%s has %d NAs", col_name, sum(is.na(col)))
}, example_features, names(example_features))

sprintf("%s has %d NAs", 'beauty', sum(is.na(outcome_buffer)))
table(outcome_buffer)

# are there any NA's in the nonbuffer DEM?
nobuffer_DEM <- withbuffer_features[!bufferzone_idx, 'dem_1']
table(is.na(nobuffer_DEM))
nobuffer_DEM_NAs_coords <- withbuffer_coords[!(bufferzone_idx) & is.na(withbuffer_features$dem_1), ]


library(sf)
# Create an sf object from the data.frame
bufferzone_sf <- st_as_sf(bufferzone, coords = c("x_coord", "y_coord"), crs = 4647)
bufferzone_sf$dem <- example_features$dem_1
bufferzone_sf$missing_DEM <- is.na(example_features$dem_1)

st_write(bufferzone_sf, "./data/models/__extracted_points/DE/beauty/all_pixels/bufferzone.geojson")


# same but for the missing DEM values
DEM_NAs <- st_as_sf(nobuffer_DEM_NAs_coords, coords = c("x_coord", "y_coord"), crs = 4647)
st_write(DEM_NAs, "./data/cleaned/dem/missing_DEM_withinEU.geojson")
