import numpy as np
import netCDF4 as nc
import scipy.io as scio
import mat73 as mat
import os
from NA_Mask_func_pkg.utils import cropped_data_outdir, cropped_data_indir

def load_mask_index_files():
    indir = '/my-projects/mask/NA_Masks/mask_index_files/'
    LANDigIND_0p01_infile = indir + 'LANDigIND_0p01.npy'
    LANDigLAT_0p01_infile = indir + 'LANDigLAT_0p01.npy'
    LANDigLON_0p01_infile = indir + 'LANDigLON_0p01.npy'
    LANDigIND_0p01 = np.load(LANDigIND_0p01_infile)
    LANDigLAT_0p01 = np.load(LANDigLAT_0p01_infile)
    LANDigLON_0p01 = np.load(LANDigLON_0p01_infile)
    return LANDigIND_0p01, LANDigLAT_0p01, LANDigLON_0p01

def load_GL_GeoLatLon():
    indir = '/my-projects/Projects/MLCNN_PM25_2021/data/'
    lat_infile = indir + 'tSATLAT.npy'
    lon_infile = indir + 'tSATLON.npy'
    GL_GeoLAT = np.load(lat_infile)
    GL_GeoLON = np.load(lon_infile)
    return GL_GeoLAT, GL_GeoLON

def load_NA_GeoLatLon():
    indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/'
    lat_infile = indir + 'tSATLAT_NA.npy'
    lon_infile = indir + 'tSATLON_NA.npy'
    NA_GeoLAT = np.load(lat_infile)
    NA_GeoLON = np.load(lon_infile)
    return NA_GeoLAT, NA_GeoLON

def load_initial_mask(Area_Name:str,region_type_name:str):
    indir = '/my-projects/mask/NA_Masks/'
    infile = indir + '{}-{}.mat'.format(region_type_name.upper(),Area_Name)
    Mask_file = scio.loadmat(infile)
    Mask_Array = Mask_file[region_type_name.lower()][:]
    return Mask_Array

def load_cropped_mask_map(Area_Name:str,region_type_name:str):
    indir = cropped_data_indir
    infile = indir + 'Cropped_{}-{}.nc'.format(region_type_name.upper(),Area_Name)
    data = nc.Dataset(infile)
    cropped_mask_data = data[region_type_name.lower()][:]
    Lat = data['lat'][:]
    Lon = data['lon'][:]
    return cropped_mask_data, Lat, Lon

def save_cropped_mask_map(Cropped_Map_Data:np.array, Geo_lat:np.array, Geo_lon:np.array, Area_Name:str,region_type_name:str):
    outdir = cropped_data_outdir
    if not os.path.isdir(outdir): 
        os.makedirs(outdir)
    outfile = outdir + 'Cropped_{}-{}.nc'.format(region_type_name.upper(),Area_Name)
    data = nc.Dataset(outfile, 'w', format='NETCDF4')
    data.TITLE = 'Mask Map for {} - {} over North America'.format(region_type_name.upper(),Area_Name)
    data.createDimension('latitude', len(Geo_lat))
    data.createDimension('longitude', len(Geo_lon))
    data.createVariable('{}'.format(region_type_name.lower()),'f8', ('latitude','longitude'))[:] = Cropped_Map_Data
    data.createVariable('lat', 'f8', ('latitude'))[:]  = Geo_lat
    data.createVariable('lon', 'f8', ('longitude'))[:] = Geo_lon
    data.close()
    return