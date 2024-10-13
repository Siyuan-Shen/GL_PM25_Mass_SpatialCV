from NA_Mask_func_pkg.data_func import *
from NA_Mask_func_pkg.iostream import *
from NA_Mask_func_pkg.utils import *

def Convert2Cropped_MaskMap(region_type_name:str, Area_Name_list:list):
    LANDigIND_0p01, LANDigLAT_0p01, LANDigLON_0p01 = load_mask_index_files()
    GL_GeoLAT, GL_GeoLON = load_GL_GeoLatLon()
    NA_GeoLAT, NA_GeoLON = load_NA_GeoLatLon()
    for Area_Name in Area_Name_list:
        print(Area_Name)
        Mask_Array = load_initial_mask(Area_Name=Area_Name, region_type_name=region_type_name)
        Mask_Array = np.squeeze(Mask_Array)
        Mask_map   = get_regional_mask_map(regional_mask_array=Mask_Array,LANDigIND_0p01=LANDigIND_0p01,
                                           tSATLAT=GL_GeoLAT,tSATLON=GL_GeoLON)
        Cropped_Mask_Map = crop_Mapdata(Init_MapData=Mask_map,lat=GL_GeoLAT,lon=GL_GeoLON, Extent=[NA_GeoLAT[0],NA_GeoLAT[-1],NA_GeoLON[0], NA_GeoLON[-1]])
        save_cropped_mask_map(Cropped_Map_Data=Cropped_Mask_Map,Geo_lat=NA_GeoLAT, Geo_lon=NA_GeoLON,Area_Name=Area_Name, region_type_name=region_type_name)
    return