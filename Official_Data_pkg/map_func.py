import numpy as np
import os
from Official_Data_pkg.utils import *
from Official_Data_pkg.iostream import save_Official_datasets,load_Official_datasets
from Estimation_pkg.iostream import load_GeoCombinedPM25_map_data,load_estimation_map_data,load_ForcedSlope_forEstimation,load_ForcedSlopeUnity_estimation_map_data
from Estimation_pkg.utils import Extent
from Training_pkg.utils import *


def Padding_Global_MapData():
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    # Get official map data latitude and longtitude length
    padding_lat_length = round(100*(Official_Global_Mapdata_Extent[1] - Official_Global_Mapdata_Extent[0]))+1
    padding_lon_length = round(100*(Official_Global_Mapdata_Extent[3] - Official_Global_Mapdata_Extent[2]))+1

    # Initialize Annual Output directory
    Annual_outdir = Official_MapData_outdir + '{}/FineResolution/{}/Annual/'.format(Official_output_data_version,'GL')
    if not os.path.isdir(Annual_outdir):
        os.makedirs(Annual_outdir)
    
    for iyear in Padding_fine_Global_Mapdata_Years:
        # Initialize Monthly Output directory
        Monthly_outdir = Official_MapData_outdir + '{}/FineResolution/{}/Monthly/{}/'.format(Official_output_data_version,'GL',iyear)
        if not os.path.isdir(Monthly_outdir):
            os.makedirs(Monthly_outdir)

        # If we want annual average, initialize annual array
        if Padding_fine_Global_Mapdata_Annual_output_switch:
            Annual_Official_Output = np.zeros((padding_lat_length,padding_lon_length),dtype=np.float32)
        
        for imonth in MONTH:

            # load initial Geo Combined Map Estimation
            temp_map_data,lat,lon = load_GeoCombinedPM25_map_data(YYYY=iyear,MM=imonth,SPECIES=species,version=version,special_name=special_name)
            Monthly_Official_Output = np.zeros((padding_lat_length,padding_lon_length),dtype=np.float32)
            print('shape of temp_map_data: {},'.format(temp_map_data.shape))
            print('shape of Monthly_Official_Output: {},'.format(Monthly_Official_Output.shape,))
            print('shape of cropped Monthly_Official_Output: {}'.format(Monthly_Official_Output[round((Extent[0]-Official_Global_Mapdata_Extent[0])*100):(round((Extent[1]-Official_Global_Mapdata_Extent[0])*100)+1),
                                    round((Extent[2]-Official_Global_Mapdata_Extent[2])*100):(round((Extent[3]-Official_Global_Mapdata_Extent[2])*100)+1)].shape))
            # Edge padding
            Monthly_Official_Output[round((Extent[0]-Official_Global_Mapdata_Extent[0])*100):(round((Extent[1]-Official_Global_Mapdata_Extent[0])*100)+1),
                                    round((Extent[2]-Official_Global_Mapdata_Extent[2])*100):(round((Extent[3]-Official_Global_Mapdata_Extent[2])*100)+1)] = temp_map_data
            # Monthly Saving
            if Padding_fine_Global_Mapdata_Monthly_output_switch:
                monthly_outfile = Monthly_outdir + '{}.CNNPM25.GL.{}{}-{}{}.nc'.format(Official_output_data_version,iyear,imonth,iyear,imonth)
                save_Official_datasets(PM25_data=Monthly_Official_Output,outfile=monthly_outfile,extent=Official_Global_Mapdata_Extent,area='GL',year=iyear,month=imonth,
                                       Annual=False,Monthly=True,resolution=0.01)
            # recording Annual Map data
            if Padding_fine_Global_Mapdata_Annual_output_switch:
                Annual_Official_Output += Monthly_Official_Output

        # Derive and save annual output
        if Padding_fine_Global_Mapdata_Annual_output_switch:
            Annual_outfile = Annual_outdir + '{}.CNNPM25.GL.{}{}-{}{}.nc'.format(Official_output_data_version,iyear,MONTH[0],iyear,MONTH[-1])
            Annual_Official_Output /= 12.0
            save_Official_datasets(PM25_data=Annual_Official_Output,outfile=Annual_outfile,extent=Official_Global_Mapdata_Extent,area='GL',year=iyear,month=imonth,
                                        Annual=True,Monthly=False,resolution=0.01)

    return

def crop_MapData():
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    if Crop_fine_Mapdata_regions_Annual_output_switch:
        for iyear in Crop_fine_Mapdata_regions_Years:
            
            # load Annual Official GLobal Data
            Annual_indir = Official_MapData_outdir + '{}/FineResolution/{}/Annual/'.format(Official_output_data_version,'GL')
            Annual_infile = Annual_indir + '{}.CNNPM25.GL.{}{}-{}{}.nc'.format(Official_output_data_version,iyear,MONTH[0],iyear,MONTH[-1])
            Annual_mapdata, lat, lon = load_Official_datasets(infile=Annual_infile)

            for iregion in range(len(Crop_fine_Mapdata_regions)):

                # Initialize Annual Output directory
                Annual_outdir = Official_MapData_outdir + '{}/FineResolution/{}/Annual/'.format(Official_output_data_version,Crop_fine_Mapdata_regions[iregion])
                if not os.path.isdir(Annual_outdir):
                    os.makedirs(Annual_outdir)
                # Get cropping index and crop init mapdata
                lat_start_index, lon_start_index, lat_end_index, lon_end_index = derive_index_cropped_Mapdata(extent=Crop_fine_Mapdata_Extents[iregion],lat=lat,lon=lon)
                cropped_mapdata = Annual_mapdata[lat_start_index:lat_end_index+1,lon_start_index:lon_end_index+1]
                
                # Save cropped Annual mapdata
                Annual_outfile = Annual_outdir + '{}.CNNPM25.{}.{}{}-{}{}.nc'.format(Official_output_data_version,Crop_fine_Mapdata_regions[iregion],iyear,MONTH[0],iyear,MONTH[-1])
                save_Official_datasets(PM25_data=cropped_mapdata,outfile=Annual_outfile,extent=Crop_fine_Mapdata_Extents[iregion],area=Crop_fine_Mapdata_regions[iregion],year=iyear,month=MONTH[0],Annual=True,Monthly=False,resolution=0.01)
    
    if Crop_fine_Mapdata_regions_Monthly_output_switch:
        for iyear in Crop_fine_Mapdata_regions_Years:
            # load Monthly Official GLobal Data
            Monthly_indir = Official_MapData_outdir + '{}/FineResolution/{}/Monthly/{}/'.format(Official_output_data_version,'GL',iyear)
            for imonth in MONTH:
                Monthly_infile  = Monthly_indir + '{}.CNNPM25.GL.{}{}-{}{}.nc'.format(Official_output_data_version,iyear,imonth,iyear,imonth)
                Monthly_mapdata, lat, lon = load_Official_datasets(infile=Monthly_infile)
                for iregion in range(len(Crop_fine_Mapdata_regions)):

                    # Initialize Monthly Output directory
                    Monthly_outdir = Official_MapData_outdir + '{}/FineResolution/{}/Monthly/{}/'.format(Official_output_data_version,Crop_fine_Mapdata_regions[iregion],iyear)
                    if not os.path.isdir(Monthly_outdir):
                        os.makedirs(Monthly_outdir)
                    
                    # Get cropping index and crop init mapdata
                    lat_start_index, lon_start_index, lat_end_index, lon_end_index = derive_index_cropped_Mapdata(extent=Crop_fine_Mapdata_Extents[iregion],lat=lat,lon=lon)
                    cropped_mapdata = Monthly_mapdata[lat_start_index:lat_end_index+1,lon_start_index:lon_end_index+1]

                    # Save cropped Monthly mapdata
                    Monthly_outfile = Monthly_outdir + '{}.CNNPM25.{}.{}{}-{}{}.nc'.format(Official_output_data_version,Crop_fine_Mapdata_regions[iregion],iyear,imonth,iyear,imonth)
                    save_Official_datasets(PM25_data=cropped_mapdata,outfile=Monthly_outfile,extent=Crop_fine_Mapdata_Extents[iregion],area=Crop_fine_Mapdata_regions[iregion],year=iyear,month=imonth,Annual=False,Monthly=True,resolution=0.01)
        
                    

    return

def derive_index_cropped_Mapdata(extent,lat,lon):
    """This function is used to derive indices for cropping data from a larger map data.

    Args:
        extent (list): desired regional extent
        lat (np.array): lat array for large mapdata
        lon (np.array): lon array for large mapdata
    """
    # derive grids numbers in one degree
    lat_grids_numbers_4one_degree = round(1.0/round(lat[2]-lat[1],3))
    lon_grids_numbers_4one_degree = round(1.0/round(lon[2]-lon[1],3))

    # get lat lon start and end index
    lat_start_index = round((extent[0] - lat[0])* lat_grids_numbers_4one_degree )
    lon_start_index = round((extent[2] - lon[0]) * lon_grids_numbers_4one_degree )
    lat_end_index = round((extent[1] - lat[0]) * lat_grids_numbers_4one_degree )
    lon_end_index = round((extent[3] - lon[0])*lon_grids_numbers_4one_degree)
    return lat_start_index, lon_start_index, lat_end_index, lon_end_index
