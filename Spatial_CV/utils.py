import numpy as np
import toml
import torch

cfg = toml.load('./config.toml')


#######################################################################################
# Observation Path
obs_dir = cfg['Pathway']['observations-dir']

geophysical_species_data_dir = obs_dir['geophysical_species_data_dir']
geophysical_biases_data_dir  = obs_dir['geophysical_biases_data_dir']
ground_observation_data_dir  = obs_dir['ground_observation_data_dir']

#######################################################################################
# Training file Path
Training_dir = cfg['Pathway']['TrainingModule-dir']

training_infile = Training_dir['training_infile']
model_outdir = Training_dir['model_outdir']

#######################################################################################
Config_outdir = cfg['Pathway']['Config-outdir']['Config_outdir']
Loss_Accuracy_outdir = cfg['Pathway']['Figures-dir']['Loss_Accuracy_outdir']
Scatter_plots_outdir = cfg['Pathway']['Figures-dir']['Scatter_plots_outdir']
#######################################################################################
# identity settings
identity = cfg['Training-Settings']['identity']

special_name = identity['special_name']
version = identity['version']

#######################################################################################
# Hyperparameters settings
HyperParameters = cfg['Training-Settings']['hyper-parameters']
channel_index = HyperParameters['channel_index']
channel_names = HyperParameters['channel_names']
epoch = HyperParameters['epoch']
batchsize = HyperParameters['batchsize']

#######################################################################################
# learning rate settings
lr_settings = cfg['Training-Settings']['learning_rate']
lr0 = lr_settings['learning_rate0']


### Strategy
ExponentialLR = lr_settings['ExponentialLR']['Settings']
ExponentialLR_gamma = lr_settings['ExponentialLR']['gamma']

CosineAnnealingLR = lr_settings['CosineAnnealingLR']['Settings']
CosineAnnealingLR_T_max = lr_settings['CosineAnnealingLR']['T_max']
CosineAnnealingLR_eta_min = lr_settings['CosineAnnealingLR']['eta_min']
#######################################################################################
# Learning Objectives Settings
learning_objective = cfg['Training-Settings']['learning-objective']

bias = learning_objective['bias']
normalize_bias = learning_objective['normalize_bias']
normalize_species = learning_objective['normalize_species']
absolute_species = learning_objective['absolute_species']
log_species = learning_objective['log_species']

############################ Spatial Cross-Validation ################################
Combine_with_geophysical = cfg['Spatial-CrossValidation']['Combine_with_geophysical']
Spatial_CrossValidation_Switch = cfg['Spatial-CrossValidation']['Spatial_CrossValidation_Switch'] # On/Off for Spatial Crosss Validation
Spatial_CV_LossAccuracy_plot_Switch = cfg['Spatial-CrossValidation']['Spatial_CV_LossAccuracy_plot_Switch']
regression_plot_switch   = cfg['Spatial-CrossValidation']['Visualization_Settings']['regression_plot_switch']
#######################################################################################
# training Settings
Spatial_Trainning_Settings = cfg['Spatial-CrossValidation']['Training_Settings']

kfold = Spatial_Trainning_Settings['kfold']
repeats = Spatial_Trainning_Settings['repeats']
beginyears = Spatial_Trainning_Settings['beginyears']
endyears = Spatial_Trainning_Settings['endyears']
NA_beginyear = Spatial_Trainning_Settings['Area_beginyears']['NA']
AS_beginyear = Spatial_Trainning_Settings['Area_beginyears']['AS']
EU_beginyear = Spatial_Trainning_Settings['Area_beginyears']['EU']
GL_beginyear = Spatial_Trainning_Settings['Area_beginyears']['GL']
MultiyearForMultiAreasLists = Spatial_Trainning_Settings['MultiyearForMultiAreasList']
#######################################################################################
# Forced Slope Unity Settings
ForcedSlopeUnityTable = cfg['Spatial-CrossValidation']['Forced-Slope-Unity']

ForcedSlopeUnity = ForcedSlopeUnityTable['ForcedSlopeUnity']
EachMonthForcedSlopeUnity = ForcedSlopeUnityTable['EachMonthForcedSlopeUnity']

#######################################################################################
# Training file Path
results_dir = cfg['Pathway']['Results-dir'] 

txt_dir = results_dir['txt_outdir']


def lr_strategy_lookup_table(optimizer):
    if ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=ExponentialLR_gamma)
    elif CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=CosineAnnealingLR_T_max,eta_min=CosineAnnealingLR_eta_min)
    
def get_typeName():
    if bias == True:
        typeName = 'PM25Bias'
    elif normalize_bias:
        typeName = 'NormalizedPM25Bias'
    elif normalize_species == True:
        typeName = 'NormaizedPM25'
    elif absolute_species == True:
        typeName = 'AbsolutePM25'
    elif log_species == True:
        typeName = 'LogPM25'
    return typeName

def get_gpu_information():
    availability   = torch.cuda.is_available()
    devices_number = torch.cuda.device_count()
    devices_names  = torch.cuda.get_device_name(0)
    current_device = torch.cuda.current_device()
    print('GPU information: \nAvailability: {}, \ndevices numbers: {}, \ndevices names: {}, \ncurrent device: {}'.format(availability, devices_number, devices_names, current_device))
    return availability, devices_number, devices_names, current_device

def pretrained_regional_group(extent):
    input_dir = '/my-projects/Projects/MLCNN_PM25_2021/code/Data_Processing/Get_Pretrained_sites_loc/Data/'
    sitelat_infile = input_dir + 'pretrained_sites_lat_index.npy'
    sitelon_infile = input_dir + 'pretrained_sites_lon_index.npy'
    site_lon_array = np.load(sitelon_infile)
    site_lat_array = np.load(sitelat_infile)
    nsite = len(site_lon_array)

    lat_index = np.array([], dtype=int)
    lon_index = np.array([], dtype=int)
    for isite in range(nsite):
        if site_lon_array[isite] >= extent[2] and site_lon_array[isite]<= extent[3]:
            lon_index = np.append(lon_index,isite)

    #lon_index = np.array(lon_index)

    for index in range(len(lon_index)):
        if site_lat_array[lon_index[index]] >= extent[0] and site_lat_array[lon_index[index]] <= extent[1]:
            lat_index = np.append(lat_index,lon_index[index])


    region_index = lat_index

    return  region_index

def pretrained_get_area_index(extent:np.array,test_index:np.array)->np.array:
    ### Get the area_index in test index
    area_index = np.array([], dtype=int)
    for iblocks in range(len(extent)):
        temp_index = pretrained_regional_group(extent=extent[iblocks])
        area_index = np.append(area_index, temp_index)
    area_index = np.intersect1d(area_index,test_index)
    return area_index

def regional_group(extent):
    input_dir = '/my-projects/Projects/MLCNN_PM25_2021/data/'
    sitelat_infile = input_dir + 'sitelat.npy'
    sitelon_infile = input_dir + 'sitelon.npy'
    site_lon_array = np.load(sitelon_infile)
    site_lat_array = np.load(sitelat_infile)
    nsite = len(site_lon_array)

    lat_index = np.array([], dtype=int)
    lon_index = np.array([], dtype=int)
    for isite in range(nsite):
        if site_lon_array[isite] >= extent[2] and site_lon_array[isite]<= extent[3]:
            lon_index = np.append(lon_index,isite)

    #lon_index = np.array(lon_index)

    for index in range(len(lon_index)):
        if site_lat_array[lon_index[index]] >= extent[0] and site_lat_array[lon_index[index]] <= extent[1]:
            lat_index = np.append(lat_index,lon_index[index])


    region_index = lat_index

    return  region_index



def extent_table() -> dict:
    
    '''
    dic = {
        'GL': [[-60.0,70.0,-179.95,179.95]],
        'NA': [[25.0, 70.0, -179.90, -33.0]],
        'SA': [[-60.00, 25.0, -179.90, -33.0]],
        'EU': [[35.0, 70.0, -33.0, 33.0]],
        'AF': [[-60.0, 35.0, -39.0, 61.0]],
        'AS': [[0.0, 70.00, 61.0, 179.90]],
        'AU': [[-60.0, 0.0, 61.0, 179.90]],
    }
    '''
    dic = {
        'GL': [[-60.0,70.0,-179.95,179.95]],
        'NA': [[25.0, 55.0, -135.0, -65.0]],
        'SA': [[-60.00, 25.0, -179.90, -33.0]],
        'EU': [[35.0, 60.0, -33.0, 35.0]],
        'AF': [[-60.0, 35.0, -39.0, 61.0]],
        'AS': [[-10.0, 45.00, 45.0, 145]],
        'AU': [[-60.0, 0.0, 61.0, 179.90]],
    }

    return dic

def get_area_index(extent:np.array,test_index)->np.array:
    ### Get the area_index in test index
    area_index = np.array([], dtype=int)
    for iblocks in range(len(extent)):
        temp_index = regional_group(extent=extent[iblocks])
        area_index = np.append(area_index, temp_index)
    area_index = np.intersect1d(area_index,test_index)
    return area_index

def get_test_index_inGBD_area(GBD_area_index,test_index):
    area_index = np.intersect1d(GBD_area_index,test_index)
    return area_index
def load_GBD_area_index(area:str):
    indir = '/my-projects/Projects/MLCNN_PM25_2021/code/Compare_gridded_PM25/data/GBD_area_index/'
    infile = indir + area +'_area_index.npy'
    GBD_area_index = np.load(infile)
    return GBD_area_index

def get_nearest_test_distance(area_test_index,area_train_index):
    """This function is used to calcaulate the nearest distance from one site in 
    testing datasets to the whole training datasets.

    Args:
        area_test_index (numpy): Testing index
        area_train_index (numpy): Training index
    return: nearest distances for testing datasets. len(area_test_index)
    """
    input_dir = '/my-projects/Projects/MLCNN_PM25_2021/data/'
    sitelat_infile = input_dir + 'sitelat.npy'
    sitelon_infile = input_dir + 'sitelon.npy'
    site_lon = np.load(sitelon_infile)
    site_lat = np.load(sitelat_infile)
    nearest_site_distance = np.full((len(area_test_index)),-999.99)
    for index in range(len(area_test_index)):
        temp_lat, temp_lon = site_lat[area_test_index[index]], site_lon[area_test_index[index]]
        other_sites_distances = calculate_distance_forArray(site_lat=temp_lat,site_lon=temp_lon,
                                                            SATLAT_MAP=site_lat[area_train_index],SATLON_MAP=site_lon[area_train_index])
        nearest_site_distance[index] = min(other_sites_distances[np.where(other_sites_distances>0.0)]) # We take 110 kilometers for one degree
    
    return nearest_site_distance
def calculate_distance(pixel_lat:np.float32,pixel_lon:np.float32,site_lat:np.float32,site_lon:np.float32,r=6371.01):
    site_pos1 = pixel_lat * np.pi / 180.0
    site_pos2 = pixel_lon * np.pi / 180.0
    other_sites_pos1_array = site_lat * np.pi / 180.0
    other_sites_pos2_array = site_lon * np.pi / 180.0
    dist = r * np.arccos(np.sin(site_pos1)*np.sin(other_sites_pos1_array)+np.cos(site_pos1)*np.cos(other_sites_pos1_array)*np.cos(site_pos2-other_sites_pos2_array))
    return dist

def calculate_distance_forArray(site_lat:np.float32,site_lon:np.float32,
                                SATLAT_MAP:np.array,SATLON_MAP:np.array,r=6371.01):
    site_pos1 = site_lat * np.pi / 180.0
    site_pos2 = site_lon * np.pi / 180.0
    other_sites_pos1_array = SATLAT_MAP * np.pi / 180.0
    other_sites_pos2_array = SATLON_MAP * np.pi / 180.0
    dist_map = r * np.arccos(np.sin(site_pos1)*np.sin(other_sites_pos1_array)+np.cos(site_pos1)*np.cos(other_sites_pos1_array)*np.cos(site_pos2-other_sites_pos2_array))
    return dist_map

def get_coefficients(nearest_site_distance,beginyear,endyear):
    """This function is used to calculate the coefficient of the combine with Geophysical PM2.5

    Args:
        nearest_site_distance (_type_): _description_
        beginyear (_type_): _description_
        endyear (_type_): _description_

    Returns:
        _type_: _description_
    """
    coefficient = (nearest_site_distance - 150.0)/(nearest_site_distance+0.0000001)
    coefficient[np.where(coefficient<0.0)]=0.0
    coefficient = np.square(coefficient)
    coefficients = np.zeros((12 * (endyear - beginyear + 1) * len(nearest_site_distance)), dtype=int)  
    for i in range(12 * (endyear - beginyear + 1)):  
        coefficients[i * len(nearest_site_distance):(i + 1) * len(nearest_site_distance)] = coefficient
    
    return coefficients
    
