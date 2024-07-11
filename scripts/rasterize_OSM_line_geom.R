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

# Define your own raster specifications
xmin <- 32280000
xmax <- 32922000
ymin <- 5235000
ymax <- 6102000
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
y <- rasterizeGeom(v, r, "length")

# Save the resulting raster to a file
writeRaster(y, output_path, overwrite = TRUE)

cat("Rasterization complete. Output saved to:", output_path, "\n")
