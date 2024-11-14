import numpy as np
import netCDF4 as nc

def load_Official_datasets(infile):
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables['PM25'][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def save_Official_datasets(PM25_data:np.array,outfile:str,extent:np.array,area:str,year:str,month:str,Annual:bool,Monthly:bool,resolution:float):

    if Annual == True:
        timescale = 'Annual'
        timecoverage = str(year)
    elif Monthly == True:
        timescale = 'Monthly'
        timecoverage = str(year) + str(month)

    lat_size = PM25_data.shape[0]
    lon_size = PM25_data.shape[1]
    lat_delta = resolution #(extent[1]-extent[0])/(lat_size-1)
    lon_delta = resolution #(extent[3]-extent[2])/(lon_size-1)
    complevel = 5
    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Convolutional Neural Network {} PM2.5 Estimation over {} Area. ({}x{} resolution)'.format(timescale, area, resolution,resolution)
    MapData.CONTACT = 'SIYUAN SHEN <s.siyuan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.SPATIALCOVERAGE = area
    MapData.TIMECOVERAGE    = timecoverage
    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    PM25 = MapData.createVariable('PM25','f4',('lat','lon',),compression= 'zlib',complevel=complevel)
    latitudes = MapData.createVariable("lat","f4",("lat",),compression= 'zlib',complevel=complevel)
    longitudes = MapData.createVariable("lon","f4",("lon",),compression= 'zlib',complevel=complevel)

    print('lat size: {}, lon size: {}, \nround((extent[1]-extent[0]+lat_delta)/lat_delta):{}, \nround((extent[3]-extent[2]+lon_delta)/lon_delta): {}'.format(lat_size,lon_size,round((extent[1]-extent[0]+lat_delta)/lat_delta),round((extent[3]-extent[2]+lon_delta)/lon_delta)))#,np.linspace(extent[2],extent[3],int((extent[3]-extent[2]+lon_delta)/lon_delta)))
    latitudes[:] = np.linspace(extent[0],extent[1],round((extent[1]-extent[0]+lat_delta)/lat_delta))
    longitudes[:] = np.linspace(extent[2],extent[3],round((extent[3]-extent[2]+lon_delta)/lon_delta))
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    latitudes.axis = 'X'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    longitudes.axis = 'Y'
    PM25.units = 'ug/m3'
    PM25.long_name = 'Convolutional Neural Network derived {} PM2.5 [ug/m^3]'.format(timescale)
    PM25[:] = PM25_data
    return