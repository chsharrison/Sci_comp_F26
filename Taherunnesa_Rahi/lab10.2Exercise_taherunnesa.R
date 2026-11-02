library(terra)
library(bDSSDMTools)
library(ggplot2)
library(sf)

#corals
#load a vector defining global landmasses
landmass <- vect("C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Landmass/Landmass.shp")
#plot landmass
plot(landmass, col = "#D4D4D4", border = "#D4D4D4", axes = TRUE) #hex color codes; col = color and border = border line color
#Corals occurrence (Source: UNEP-WCMC)
presences <- read.csv("C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Corals/occurrenceRecords.csv")
#transform data.frame into a spatial object
presences <- vect(presences, geom = c("Lon", "Lat"))
#plot occurrences
plot(presences, col = "#034B03", pch = 16, cex = 0.4) #pch = plotting character & cex = size

#Plot records on landmass
plot(landmass, main = "Corals Species Records", col = "#D4D4D4", border = "#D4D4D4", axes = TRUE)
plot(presences, col = "#034B03", pch = 16, cex = 0.4, add = TRUE)


#import environmental data

files <- c("C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Present Benthic Salinity Mean.tif",
           "C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Present Benthic OceanTemperature LtMax.tif",
           "C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Present Benthic OceanTemperature LtMin.tif",
           "C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Present Benthic pH Mean.tif",
           "C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Present Benthic TotalPrimaryProductionPhyto Mean.tif",
           "C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Present Surface OceanTemperature LtMax.tif",
           "C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Present Surface OceanTemperature LtMin.tif")


# produce and plot raster layers
environmentalLayers <- rast(files)
plot(environmentalLayers)

#Defining the extent as study area
ct_extent <- ext(-100, -55, 0, 32)   # xmin, xmax, ymin, ymax
# Crop presence points to IWP
presences_ct <- crop(presences, ct_extent)
plot(presences_ct)

#Look at where it is on the landmass
plot(landmass, main = "Corals Species Records", col = "#D4D4D4", border = "#D4D4D4", axes = TRUE)
plot(presences_ct, col = "#034B03", pch = 16, cex = 0.4, add = TRUE)

#Develop study region area
myStudyRegion <- studyRegion(rasterLayers = environmentalLayers,
                             presences = presences_ct,
                             distanceThreshold = 200000, #distance from records in meters
                             bathymetryLayer = "C:/Users/Taherunnesa Rahi/OneDrive - Louisiana State University/scientific computing/lab10.2/Sci_Comp26_SDM_Lab/Sci_Comp26_SDM_Lab/Sci_Comp26/Data/Raster_data/Bathymetry.tif",
                             depthPref = c(0, -60)) # To cover depth: 0 to 40 m for reef-building corals



# inspect the study region layer
plot(myStudyRegion, main="Study region", col="#979797", axes=TRUE)


# crop the environmental layers to the extent of the study region
environmentalLayers <- crop(environmentalLayers, myStudyRegion)
# mask the environmental layers with the study region
environmentalLayers <- mask(environmentalLayers, myStudyRegion)
# plot the masked environmental layers
plot(environmentalLayers, axes = TRUE)



# generate pseudo-absences
absences <- pseudoAbsences(rasterLayers = environmentalLayers, presences = presences_ct, n = 2000)
# crop landmass to the region for better representation
landmassCT <- crop(landmass, myStudyRegion)
# plot records
plot(landmassCT, main = "Coral Triangle Species records", col = "#D4D4D4", border = "#D4D4D4", axes = TRUE)
plot(absences, col = "#c29431", pch = 16, cex = 0.5, add = TRUE)
plot(presences_ct, col = "#043259", pch = 16, cex = 0.5, add = TRUE)

# combine presences and absences into a unique object for modelling
presences_ct$PA <- 1
absences$PA <- 0
records <- rbind(presences_ct, absences)
# plot records
plot(landmassCT, main = "Coral Reefs Species records", col = "#D4D4D4", border = "#D4D4D4", axes = TRUE)
plot(records, y = "PA", col = c("#c29431", "#043259"), pch = 16, cex = 0.5, add = TRUE)

# extract environmental values and make an object with all information to model: Model evaluation with cross validation
modelData <- generateModelData(records, envConditions = environmentalLayers, method = "blocks", kFolds = 10)

# inspect the data partitioning
modelData$plotDatasets

# fit a BRT model to the dataset
model <- trainModel(modelData, algorithm = "brt")

# predict the distribution with the model over a raster stack
prediction <- predictModel(model = model, newData = environmentalLayers)

# map the predicted distribution
plot(prediction$rasterLayer, main = "Predicted Species distribution", col = rev(topo.colors(100)))




