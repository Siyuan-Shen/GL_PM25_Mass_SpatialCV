import toml
import numpy as np
import time
import math

cfg = toml.load('./config.toml')

#######################################################################################
# outdir
Official_MapData_outdir = cfg['Pathway']['Estimation-dir']['Official_MapData_outdir']

############################ OfficialData Settings ################################

Derive_OfficialData_Switch = cfg['OfficialData-Settings']['Derive_OfficialData_Switch'] 
Padding_fine_Global_Mapdata_Switch = cfg['OfficialData-Settings']['Padding_fine_Global_Mapdata_Switch']
Crop_fine_Mapdata_regions_Switch = cfg['OfficialData-Settings']['Crop_fine_Mapdata_regions_Switch']
Plot_OfficialData_Switch = cfg['OfficialData-Settings']['Plot_OfficialData_Switch']
Convert_fine2coarse_Mapdata_Switch = cfg['OfficialData-Settings']['Convert_fine2coarse_Mapdata_Switch']
Official_output_data_version = cfg['OfficialData-Settings']['Official_output_data_version']
Use_ForcedSlopeUnity_Switch = cfg['OfficialData-Settings']['Use_ForcedSlopeUnity_Switch']
############################ Padding Settings ################################

Padding_fine_Global_Mapdata_Settings = cfg['OfficialData-Settings']['Padding_fine_Global_Mapdata'] 
Padding_fine_Global_Mapdata_Years = Padding_fine_Global_Mapdata_Settings['Padding_fine_Global_Mapdata_Years']
Padding_fine_Global_Mapdata_Annual_output_switch  = Padding_fine_Global_Mapdata_Settings['Annual_output_switch']
Padding_fine_Global_Mapdata_Monthly_output_switch = Padding_fine_Global_Mapdata_Settings['Monthly_output_switch']
Official_Global_Mapdata_Extent = Padding_fine_Global_Mapdata_Settings['Official_Global_Mapdata_Extent']

############################ Crop fine Map Data Settings ################################

Crop_fine_Mapdata_regions_Settings = cfg['OfficialData-Settings']['Crop_fine_Mapdata_regions'] 
Crop_fine_Mapdata_regions_Years = Crop_fine_Mapdata_regions_Settings['Crop_fine_Mapdata_regions_Years']
Crop_fine_Mapdata_regions_Annual_output_switch  = Crop_fine_Mapdata_regions_Settings['Annual_output_switch']
Crop_fine_Mapdata_regions_Monthly_output_switch = Crop_fine_Mapdata_regions_Settings['Monthly_output_switch']
Crop_fine_Mapdata_regions = Crop_fine_Mapdata_regions_Settings['Crop_fine_Mapdata_regions']
Crop_fine_Mapdata_Extents = Crop_fine_Mapdata_regions_Settings['Crop_fine_Mapdata_Extents']

############################ Convert fine to coarse Map Data Settings ################################

Convert_fine2coarse_Mapdata_Settings = cfg['OfficialData-Settings']['Convert_fine2coarse_Mapdata'] 
Convert_fine2coarse_Mapdata_Years = Convert_fine2coarse_Mapdata_Settings['Convert_fine2coarse_Mapdata_Years']
Convert_fine2coarse_Mapdata_Annual_output_switch  = Convert_fine2coarse_Mapdata_Settings['Annual_output_switch']
Convert_fine2coarse_Mapdata_Monthly_output_switch = Convert_fine2coarse_Mapdata_Settings['Monthly_output_switch']
Convert_Coarse_Mapdata_regions = Convert_fine2coarse_Mapdata_Settings['Convert_Coarse_Mapdata_regions']
Convert_Coarse_Mapdata_Extents = Convert_fine2coarse_Mapdata_Settings['Convert_Coarse_Mapdata_Extents']


################################ Plot Official Data Settings ################################

Plot_OfficialData_Settings = cfg['OfficialData-Settings']['Plot_OfficialData']
Annual_plot_Switch = Plot_OfficialData_Settings['Annual_plot_Switch']
Monthly_plot_Switch = Plot_OfficialData_Settings['Monthly_plot_Switch']
Plot_OfficialData_Years = Plot_OfficialData_Settings['Plot_OfficialData_Years']
Plot_OfficialData_Months = Plot_OfficialData_Settings['Plot_OfficialData_MONTHS']
Plot_OfficialData_Area  = Plot_OfficialData_Settings['Plot_OfficialData_Area']
Plot_OfficialData_Extent = Plot_OfficialData_Settings['Plot_OfficialData_Extent']
