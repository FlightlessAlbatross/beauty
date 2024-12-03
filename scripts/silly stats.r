#silly stats


# calculate the most beautyful country. 
library(terra)

EU_countries <- vect('data/cleaned/NUTS/EU_mainland_countries.geojson')
beauty <- rast('data/models/DE__beauty__random_pixels__XGB__asis__7_021224/prediction.tif')

country_beauty <- data.frame(ID = EU_countries$NUTS_ID  ,
                            country = EU_countries$NAME_LATN  ,
                             zonal(beauty, EU_countries, fun = 'mean') )

country_beauty <- country_beauty[order(country_beauty$prediction),]
country_beauty
