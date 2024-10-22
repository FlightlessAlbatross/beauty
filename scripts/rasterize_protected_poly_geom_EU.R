#!/usr/bin/env Rscript

# Load required library
library(terra)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if correct number of arguments is provided
if (length(args) != 2) {
  stop("Please provide input vector path and output raster path as arguments.")
}

input_path <- args[1]
output_path <- args[2]

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
r <- rast(ncol = (xmax - xmin) / resolution, 
          nrow = (ymax - ymin) / resolution, 
          xmin = xmin, xmax = xmax, 
          ymin = ymin, ymax = ymax, 
          crs = crs)

# Load the vector data
v <- vect(input_path)

# Rasterize the vector data onto the raster using the "length" attribute
y <- rasterizeGeom(v, r, "area")

# Save the resulting raster to a file
writeRaster(y, output_path, overwrite = TRUE)

if (file.exists(output_path)) {
  cat("Rasterization complete. Output saved to:", output_path, "\n")
} else {
  cat("unknown error no output created", "\n")
}
