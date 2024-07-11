# Load required libraries
library(optparse)
library(sf)
library(terra)



# Define command-line arguments
option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL,
              help = "Input directory with shapefiles", metavar = "character"),
  make_option(c("-o", "--output"), type = "character", default = NULL,
              help = "Output directory for raster files", metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Check if input and output directories are provided
if (is.null(opt$input) | is.null(opt$output)) {
  print_help(opt_parser)
  stop("Both input and output directories must be specified.", call. = FALSE)
}

# Set input and output directories
input_dir <- opt$input
output_dir <- opt$output

# Get the list of shapefiles
filelist <- list.files(path = input_dir, pattern = '.shp$', full.names = TRUE)

if (length(filelist) == 0){
    print("No .shp files found at this input directory")
}

make_raster <- function(polypath) {
  # Extract extent and CRS from sf object
  outfile <- file.path(output_dir, gsub(pattern = 'shape.shp', replacement = 'DE.tif', basename(polypath)))
  

    poly <- st_read(polypath, quiet = TRUE)

  
  
  extent <- as.vector(st_bbox(poly))
  crs <- st_crs(poly)$wkt # or use st_crs(polygons)$epsg if you prefer EPSG codes
  
  # Create an empty raster with the desired resolution
  # Calculate the number of rows and columns based on the resolution
  xres <- 1000  # resolution in meters
  ncol <- ceiling((extent[3] - extent[1]) / xres)
  nrow <- ceiling((extent[4] - extent[2]) / xres)
  
  raster_template <- rast(
    xmin = extent[1],
    xmax = extent[3],
    ymin = extent[2],
    ymax = extent[4],
    nrows = nrow,
    ncols = ncol,
    crs = crs
  )
  
  # Rasterize the polygon
  rasterized_poly <- rasterize(poly, raster_template, field = 'gridcode')
  
  writeRaster(
    rasterized_poly,
    filename = outfile,
    datatype = 'INT1U',
    gdal = c("COMPRESS=DEFLATE", "TFW=YES"),
    overwrite = TRUE
  )
}

invisible(lapply(filelist, make_raster))
