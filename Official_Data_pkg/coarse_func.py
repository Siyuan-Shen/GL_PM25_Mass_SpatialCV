import numpy as np
import netCDF4 as nc
import os
from Training_pkg.utils import *
from Official_Data_pkg.utils import *
from Official_Data_pkg.iostream import load_Official_datasets,save_Official_datasets

def Convert_mapdata_fine2coarse_resolution():
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    if Convert_fine2coarse_Mapdata_Annual_output_switch:
        for iyear in Convert_fine2coarse_Mapdata_Years:
            for iregion in range(len(Convert_Coarse_Mapdata_regions)):
                # load Annual Official Regional Data with fine resolution
                Annual_indir = Official_MapData_outdir + '{}/FineResolution/{}/Annual/'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion])
                Annual_infile = Annual_indir + '{}.CNNPM25.{}.{}{}-{}{}.nc'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion],iyear,MONTH[0],iyear,MONTH[-1])
                Annual_mapdata, lat, lon = load_Official_datasets(infile=Annual_infile)

                # Convert to coarse resoltution
                Coarse_PM25_Map, Coarse_lat, Coarse_lon = convert_Fine2Coarse_Resolution(Convert_Coarse_Mapdata_Extents[iregion],Annual_mapdata)

                # Save Converted Mapdata
                Annual_outdir = Official_MapData_outdir + '{}/CoarseResolution/{}/Annual/'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion])
                if not os.path.isdir(Annual_outdir):
                    os.makedirs(Annual_outdir)
                Annual_outfile = Annual_outdir + '{}.CNNPM25.0p10.{}.{}{}-{}{}.nc'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion],iyear,MONTH[0],iyear,MONTH[-1])

                save_Official_datasets(PM25_data=Coarse_PM25_Map,outfile=Annual_outfile,extent=Convert_Coarse_Mapdata_Extents[iregion],area=Convert_Coarse_Mapdata_regions[iregion],
                                       year=iyear,month=MONTH[0],Annual=True,Monthly=False,resolution=0.1)
    if Convert_fine2coarse_Mapdata_Monthly_output_switch:
        for iyear in Convert_fine2coarse_Mapdata_Years:
            for imonth in MONTH:
                for iregion in range(len(Convert_Coarse_Mapdata_regions)):
                    # load Monthly Official Regional Data with fine resolution
                    Monthly_indir = Official_MapData_outdir + '{}/FineResolution/{}/Monthly/{}/'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion],iyear)
                    Monthly_infile = Monthly_indir + '{}.CNNPM25.{}.{}{}-{}{}.nc'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion],iyear,imonth,iyear,imonth)
                    Monthly_mapdata, lat, lon = load_Official_datasets(infile=Monthly_infile)

                    # Convert to coarse resoltution
                    Coarse_PM25_Map, Coarse_lat, Coarse_lon = convert_Fine2Coarse_Resolution(Convert_Coarse_Mapdata_Extents[iregion],Monthly_mapdata)

                    # Save Converted Mapdata
                    Monthly_outdir = Official_MapData_outdir + '{}/CoarseResolution/{}/Monthly/{}/'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion],iyear)
                    if not os.path.isdir(Monthly_outdir):
                        os.makedirs(Monthly_outdir)
                    Monthly_outfile = Monthly_outdir + '{}.CNNPM25.0p10.{}.{}{}-{}{}.nc'.format(Official_output_data_version,Convert_Coarse_Mapdata_regions[iregion],iyear,imonth,iyear,imonth)

                    save_Official_datasets(PM25_data=Coarse_PM25_Map,outfile=Monthly_outfile,extent=Convert_Coarse_Mapdata_Extents[iregion],area=Convert_Coarse_Mapdata_regions[iregion],
                                        year=iyear,month=imonth,Annual=False,Monthly=True,resolution=0.1)

    return

def convert_Fine2Coarse_Resolution(Coarse_extent,fine_PM25_map):
    """This is used to convert mapdata from 0.01x0.01 to 0.1x0.1 resolution

    Args:
        Coarse_extent (list): extent for coarse data
        fine_PM25_map (np array): mapdata with fine resolution

    Returns:
        np.array: Coarse map and corresponding lat/lon
    """
    nlat = 1+round((Coarse_extent[1]-Coarse_extent[0])/0.1)
    nlon = 1+round((Coarse_extent[3]-Coarse_extent[2])/0.1)
    Coarse_lat = np.linspace(Coarse_extent[0],Coarse_extent[1],nlat)
    Coarse_lon = np.linspace(Coarse_extent[2],Coarse_extent[3],nlon)
    Coarse_PM25_Map = np.full((len(Coarse_lat),len(Coarse_lon)),-999.0)
    for ix in range(nlat):
        for iy in range(nlon):
            temp_area = fine_PM25_map[ix*10:(ix+1)*10,iy*10:(iy+1)*10]
            # Identify pixels with half pixels larger than 0, mostly to identify land/waterbody boundary
            if len(np.where(temp_area>0)[0])> 25:
                Coarse_PM25_Map[ix,iy] = np.average(temp_area[np.where(temp_area>0)])

    return Coarse_PM25_Map, Coarse_lat, Coarse_lon