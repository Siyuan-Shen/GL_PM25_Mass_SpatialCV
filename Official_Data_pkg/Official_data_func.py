from Official_Data_pkg.map_func import Padding_Global_MapData, crop_MapData
from Official_Data_pkg.coarse_func import Convert_mapdata_fine2coarse_resolution
from Official_Data_pkg.utils import *

def derive_official_mapdata():
    if Padding_fine_Global_Mapdata_Switch:
        Padding_Global_MapData()
    if Crop_fine_Mapdata_regions_Switch:
        crop_MapData()
    if Convert_fine2coarse_Mapdata_Switch:
        Convert_mapdata_fine2coarse_resolution()