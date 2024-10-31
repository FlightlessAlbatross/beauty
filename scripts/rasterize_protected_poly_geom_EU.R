#!/usr/bin/env Rscript

# Load required library
library(terra)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if correct number of arguments is provided
if (length(args) != 3) {
  stop("args: <input_vector> <intermediary_area_raster> <scaled_area_raster>")
}

input_path <- args[1]
intermediary_path <-  args[2]
output_path <- args[3]

if (!file.exists(input_path)) {
  stop("Input file does not exist")
}

# Define your own raster specifications
xmin <- 29172876.762312
xmax <- 35737830.291022
ymin <- 3927473.808968
ymax <- 7986389.044350
resolution <- 1000  # Pixel size
crs <- "EPSG:4647"


# Create an empty raster with the specified extent, resolution, and CRS
if (!file.exists(intermediary_path)) {
  r <- rast(ncol = (xmax - xmin) / resolution,
            nrow = (ymax - ymin) / resolution,
            xmin = xmin, xmax = xmax,
            ymin = ymin, ymax = ymax,
            crs = crs)

  # Load the vector data
  v <- vect(input_path)
  print(paste0("Loaded OSM vector data:", input_path))

  # Rasterize the vector data onto the raster using the "length" attribute

  y <- rasterizeGeom(v, r, "area")
  writeRaster(y, intermediary_path)
  print("Rasterizing done")
} else {
  print("Intermediary file exists. Loading....")
  y <- rast(intermediary_path)
}

max_area <- as.numeric(global(y, max, na.rm = TRUE))
print(paste0("The maximum length of any pixel is ", max_area))


y_norm <- y / max_area
print("Scaling done. Write to disc....")

# Save the resulting raster to a file
writeRaster(y_norm, output_path)

if (file.exists(output_path)) {
  cat("Rasterization and scaling complete. Output saved to:", output_path, "\n")
} else {
  cat("unknown error no output created", "\n")
}