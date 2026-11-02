# Lab 10.2: BRT model for Tropical Americas and Caribbean

# 1. Load packages
library(terra)
library(bDSSDMTools)
library(ggplot2)
library(sf)

# 2. Set data directory
data_dir <- "E:/_MS LSU/OCS 7001 Scientific Computing/Sci_Comp26/Sci_Comp26/Data"

# 3. Load landmass
landmass <- vect(
  file.path(data_dir, "Landmass", "Landmass.shp")
)

plot(
  landmass,
  col = "#D4D4D4",
  border = "#D4D4D4",
  axes = TRUE
)

# 4. Load coral occurrence records
presences <- read.csv(
  file.path(data_dir, "Corals", "occurrenceRecords.csv")
)

# Convert the data frame into a spatial object
presences <- vect(
  presences,
  geom = c("Lon", "Lat"),
  crs = "EPSG:4326"
)

# Plot occurrence records
plot(
  landmass,
  main = "Coral Species Records",
  col = "#D4D4D4",
  border = "#D4D4D4",
  axes = TRUE
)

plot(
  presences,
  col = "#034B03",
  pch = 16,
  cex = 0.4,
  add = TRUE
)

# 5. Import environmental data
files <- file.path(
  data_dir,
  "Raster_data",
  c(
    "Present Benthic OceanTemperature LtMax.tif",
    "Present Benthic OceanTemperature LtMin.tif",
    "Present Benthic pH Mean.tif",
    "Present Benthic Salinity Mean.tif",
    "Present Surface OceanTemperature LtMax.tif",
    "Present Surface OceanTemperature LtMin.tif"
  )
)

environmentalLayers <- rast(files)

plot(environmentalLayers)

# 6. Define the study extent
# xmin, xmax, ymin, ymax
caribbean_extent <- ext(-100, -55, 0, 32)

# Crop presence records to the Caribbean
presences_caribbean <- crop(
  presences,
  caribbean_extent
)

plot(presences_caribbean)

# Show the selected records on the landmass
plot(
  landmass,
  main = "Coral Records: Tropical Americas and Caribbean",
  col = "#D4D4D4",
  border = "#D4D4D4",
  axes = TRUE
)

plot(
  presences_caribbean,
  col = "#034B03",
  pch = 16,
  cex = 0.4,
  add = TRUE
)

# 7. Develop the study region
myStudyRegion <- studyRegion(
  rasterLayers = environmentalLayers,
  presences = presences_caribbean,
  distanceThreshold = 200000,
  bathymetryLayer = file.path(
    data_dir,
    "Raster_data",
    "Bathymetry.tif"
  ),
  depthPref = c(0, -60)
)

# Keep the study region within the exercise's extent
myStudyRegion <- crop(
  myStudyRegion,
  caribbean_extent
)

plot(
  myStudyRegion,
  main = "Study Region",
  col = "#979797",
  axes = TRUE
)

# 8. Crop and mask the environmental layers
environmentalLayers <- crop(
  environmentalLayers,
  myStudyRegion
)

environmentalLayers <- mask(
  environmentalLayers,
  myStudyRegion
)

plot(
  environmentalLayers,
  axes = TRUE
)

# 9. Generate pseudo-absences
set.seed(42)

absences <- pseudoAbsences(
  rasterLayers = environmentalLayers,
  presences = presences_caribbean,
  n = 2000
)

# Crop landmass for plotting
landmass_caribbean <- crop(
  landmass,
  myStudyRegion
)

# Plot presence and pseudo-absence records
plot(
  landmass_caribbean,
  main = "Caribbean Coral Species Records",
  col = "#D4D4D4",
  border = "#D4D4D4",
  axes = TRUE
)

plot(
  absences,
  col = "#C29431",
  pch = 16,
  cex = 0.5,
  add = TRUE
)

plot(
  presences_caribbean,
  col = "#043259",
  pch = 16,
  cex = 0.5,
  add = TRUE
)

# 10. Combine presence and pseudo-absence records
presences_caribbean$PA <- 1
absences$PA <- 0

records <- rbind(
  presences_caribbean,
  absences
)

plot(
  landmass_caribbean,
  main = "Coral Presence and Pseudo-absence",
  col = "#D4D4D4",
  border = "#D4D4D4",
  axes = TRUE
)

plot(
  records,
  y = "PA",
  col = c("#C29431", "#043259"),
  pch = 16,
  cex = 0.5,
  add = TRUE
)

# 11. Extract environmental values and prepare model data
# Spatial block cross-validation with 10 folds
modelData <- generateModelData(
  records,
  envConditions = environmentalLayers,
  method = "blocks",
  kFolds = 10
)

# Inspect training and testing partitions
print(modelData$plotDatasets)

# 12. Fit the BRT model
model <- trainModel(
  modelData,
  algorithm = "brt"
)

# 13. Predict species distribution
prediction <- predictModel(
  model = model,
  newData = environmentalLayers
)

# 14. Map the predicted distribution
plot(
  prediction$rasterLayer,
  main = "Predicted Coral Distribution: Caribbean",
  col = rev(topo.colors(100))
)