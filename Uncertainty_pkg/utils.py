import toml
import numpy as np
cfg = toml.load('./config.toml')

#######################################################################################
# Uncertainty Settings
Uncertainty_outdir = cfg['Pathway']['Estimation-dir']['Uncertainty_outdir']

Uncertainty_Settings = cfg['Uncertainty-Settings']
Uncertainty_Switch = Uncertainty_Settings['Uncertainty_Switch']
Derive_BLISCO_LOWESS_Uncertainty_Switch = Uncertainty_Settings['Derive_BLISCO_LOWESS_Uncertainty_Switch']
Derive_distances_map_Switch    = Uncertainty_Settings['Derive_distances_map_Switch']
Derive_rRMSE_map_Switch        = Uncertainty_Settings['Derive_rRMSE_map_Switch']
Derive_absolute_Uncertainty_map_Switch   = Uncertainty_Settings['Derive_absolute_Uncertainty_map_Switch']
Uncertainty_visualization_Switch = Uncertainty_Settings['Uncertainty_visualization_Switch']

Uncertainty_BLISCO_LOWESS_Uncertainty_Settings = Uncertainty_Settings['BLISCO_LOWESS_Uncertainty_Settings']
nearby_sites_distances_mode                            = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['nearby_sites_distances_mode']
number_of_nearby_sites_forAverage                     = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['number_of_nearby_sites_forAverage']
Uncertainty_BLISCO_beginyear                          = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['BLISCO_beginyear']
Uncertainty_BLISCO_endyear                            = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['BLISCO_endyear']
Uncertainty_BLISCO_kfolds                             = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['BLISCO_kfolds']
Uncertainty_BLISCO_seeds_numbers                      = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['BLISCO_seeds_numbers']
Uncertainty_Buffer_radii_forUncertainty               = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['Buffer_radii_forUncertainty']
Max_distances_for_Bins                                = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['Max_distances_for_Bins']
Number_of_Bins                                        = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['Number_of_Bins']
Low_percentile_remove                                 = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['Low_percentile_remove']
High_percentile_remove                                = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['High_percentile_remove']
LOWESS_frac                                           = Uncertainty_BLISCO_LOWESS_Uncertainty_Settings['LOWESS_frac']
Uncertainty_Map_Estimation_Settings = Uncertainty_Settings['Map_Estimation_Settings']
Uncertainty_Estimation_years  = Uncertainty_Map_Estimation_Settings['Estimation_years']
Uncertainty_Estimation_months = Uncertainty_Map_Estimation_Settings['Estiamtion_months']



Uncertainty_Visualization_Settings = Uncertainty_Settings['Visualization_Settings']
Uncertainty_Map_plot                                  = Uncertainty_Visualization_Settings['Uncertainty_Map_plot']
Uncertainty_BLISCO_LOWESS_relationship_plot           = Uncertainty_Visualization_Settings['Uncertainty_BLISCO_LOWESS_relationship_plot']
Uncertainty_Plot_YEARS  = Uncertainty_Visualization_Settings['Uncertainty_Plot_YEARS']
Uncertainty_Plot_MONTHS = Uncertainty_Visualization_Settings['Uncertainty_Plot_MONTHS']
Uncertainty_Plot_Area   = Uncertainty_Visualization_Settings['Uncertainty_Plot_Area']
Uncertainty_Plot_Extent = Uncertainty_Visualization_Settings['Uncertainty_Plot_Extent']

def get_extent_lat_lon_map(lat_index,lon_index,SATLAT,SATLON):
    extent_lat_map = np.zeros((len(lat_index),len(lon_index)),dtype=np.float32)
    extent_lon_map = np.zeros((len(lat_index),len(lon_index)),dtype=np.float32)
    for iy in range(len(lon_index)):
        extent_lat_map[:,iy] = SATLAT[lat_index]
    for ix in range(len(lat_index)):
        extent_lon_map[ix,:] = SATLON[lon_index]
    return extent_lat_map,extent_lon_map