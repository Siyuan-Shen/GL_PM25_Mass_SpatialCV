import numpy as np
import netCDF4 as nc
from Training_pkg.utils import *



def load_TrainingVariables(nametags):
    data = nc.Dataset(training_infile,'r')
    width = np.array(data.variables['width'][:])[0]
    height = np.array(data.variables['height'][:])[0]
    total_number = np.array(data.variables['Total_number'])[0]
    sitesnumber = np.array(data.variables['sites_number'])[0]
    start_YYYY  = np.array(data.variables['start_YYYY'])[0]
    TrainingDatasets = np.zeros((total_number,len(nametags),width,height),dtype=np.float64)
    for i in range(len(nametags)):
        TrainingDatasets[:,i,:,:] = np.array(data.variables[nametags[i]][:,:,:])
    return width, height, sitesnumber,start_YYYY, TrainingDatasets

def load_monthly_obs_data(species:str):
    infile = ground_observation_data_dir + ground_observation_data_infile
    data = nc.Dataset(infile)
    SPECIES_OBS = data.variables[species][:]
    SPECIES_OBS = np.array(SPECIES_OBS)
    SPECIES_OBS = np.reshape(SPECIES_OBS.T,SPECIES_OBS.shape[0]*SPECIES_OBS.shape[1])

    lat = data.variables["latitude"][:]
    lon = data.variables["longitude"][:]
    lat = np.array(lat)
    lon = np.array(lon)

    return SPECIES_OBS, lat, lon 

def load_geophysical_species_data(species:str):
    infile = geophysical_species_data_dir + geophysical_species_data_infile
    species_monthly_data = nc.Dataset(infile)
    geophysical_species = species_monthly_data.variables[species][:]
    geophysical_species = np.array(geophysical_species)
    geophysical_species = np.reshape(geophysical_species.T, geophysical_species.shape[0]*geophysical_species.shape[1])

    latitudes = species_monthly_data.variables["latitude"][:]
    longitudes = species_monthly_data.variables["longitude"][:]
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)

    return geophysical_species, latitudes, longitudes

def load_geophysical_biases_data(species:str):
    infile = geophysical_biases_data_dir + geophysical_biases_data_infile
    species_monthly_data = nc.Dataset(infile)
    geophysical_species = species_monthly_data.variables[species][:]
    geophysical_species = np.array(geophysical_species)
    geophysical_species = np.reshape(geophysical_species.T,geophysical_species.shape[0]*geophysical_species.shape[1])

    latitudes = species_monthly_data.variables["latitude"][:]
    longitudes = species_monthly_data.variables["longitude"][:]
    return geophysical_species, latitudes, longitudes


def Learning_Object_Datasets(bias:bool,Normalized_bias:bool,Normlized_Speices:bool,Absolute_Species:bool, Log_PM25:bool, species:str):
    if bias == True:
        true_input, lat, lon  = load_geophysical_biases_data(species=species)
        mean = 0
        std = 1
        return true_input, mean, std
    elif Normalized_bias == True:
        bias_data, lat, lon  = load_geophysical_biases_data(species=species)
        bias_mean = np.mean(bias_data)
        bias_std = np.std(bias_data)
        true_input = (bias_data - bias_mean) / bias_std
        return true_input, bias_mean, bias_std
    elif Normlized_Speices == True:
        obs_data, lat, lon  = load_monthly_obs_data(species=species)
        obs_mean = np.mean(obs_data)
        obs_std = np.std(obs_data)
        true_input = (obs_data - obs_mean) / obs_std
        return true_input, obs_mean, obs_std
    elif Absolute_Species == True:
        true_input ,lat, lon  = load_monthly_obs_data(species=species)
        mean = 0
        std = 1
        return true_input, mean, std
    elif Log_PM25 == True:
        obs_data ,lat, lon  = load_monthly_obs_data(species=species)
        true_input = np.log(obs_data+1)
        mean = 0
        std = 1
        return true_input, mean, std
