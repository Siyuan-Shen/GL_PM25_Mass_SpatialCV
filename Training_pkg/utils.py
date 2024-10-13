import torch
import torch.nn as nn

import toml

cfg = toml.load('./config.toml')

#######################################################################################
# Observation Path
obs_dir = cfg['Pathway']['observations-dir']

geophysical_species_data_dir    = obs_dir['geophysical_species_data_dir']
geophysical_biases_data_dir     = obs_dir['geophysical_biases_data_dir']
ground_observation_data_dir     = obs_dir['ground_observation_data_dir']
geophysical_species_data_infile = obs_dir['geophysical_species_data_infile']
geophysical_biases_data_infile  = obs_dir['geophysical_biases_data_infile']
ground_observation_data_infile  = obs_dir['ground_observation_data_infile']
#######################################################################################
# Training file Path
Training_dir = cfg['Pathway']['TrainingModule-dir']

training_infile = Training_dir['training_infile']
model_outdir = Training_dir['model_outdir']

#######################################################################################
Config_outdir = cfg['Pathway']['Config-outdir']['Config_outdir']
#######################################################################################
# identity settings
identity = cfg['Training-Settings']['identity']

special_name = identity['special_name']
version = identity['version']

#######################################################################################
# Hyperparameters settings
HyperParameters = cfg['Training-Settings']['hyper-parameters']

channel_names = HyperParameters['channel_names']
epoch = HyperParameters['epoch']
batchsize = HyperParameters['batchsize']

#######################################################################################
# Net Structure Settings

net_structure_settings = cfg['Training-Settings']['net_structure_settings']

TwoCombineModels_Settings = net_structure_settings['TwoCombineModels']['Settings']
TwoCombineModels_Variable = net_structure_settings['TwoCombineModels']['Variable']
TwoCombineModels_threshold = net_structure_settings['TwoCombineModels']['threshold']

ResNet_setting      = net_structure_settings['ResNet']['Settings']
ResNet_Blocks       = net_structure_settings['ResNet']['Blocks']
ResNet_blocks_num   = net_structure_settings['ResNet']['blocks_num']

ResNet_MLP_setting      = net_structure_settings['ResNet_MLP']['Settings']
ResNet_MLP_Blocks       = net_structure_settings['ResNet_MLP']['Blocks']
ResNet_MLP_blocks_num   = net_structure_settings['ResNet_MLP']['blocks_num']

ResNet_Classification_Settings                        = net_structure_settings['ResNet_Classification']['Settings']
ResNet_Classification_Blocks                          = net_structure_settings['ResNet_Classification']['Blocks']
ResNet_Classification_blocks_num                      = net_structure_settings['ResNet_Classification']['blocks_num']
ResNet_Classification_left_bin                        = net_structure_settings['ResNet_Classification']['left_bin']
ResNet_Classification_right_bin                       = net_structure_settings['ResNet_Classification']['right_bin']
ResNet_Classification_bins_number                     = net_structure_settings['ResNet_Classification']['bins_number']

ResNet_MultiHeadNet_Settings                          = net_structure_settings['ResNet_MultiHeadNet']['Settings']
ResNet_MultiHeadNet_Blocks                            = net_structure_settings['ResNet_MultiHeadNet']['Blocks']
ResNet_MultiHeadNet_blocks_num                        = net_structure_settings['ResNet_MultiHeadNet']['blocks_num']
ResNet_MultiHeadNet_left_bin                          = net_structure_settings['ResNet_MultiHeadNet']['left_bin']
ResNet_MultiHeadNet_right_bin                         = net_structure_settings['ResNet_MultiHeadNet']['right_bin']
ResNet_MultiHeadNet_bins_number                       = net_structure_settings['ResNet_MultiHeadNet']['bins_number']
ResNet_MultiHeadNet_regression_portion                = net_structure_settings['ResNet_MultiHeadNet']['regression_portion']
ResNet_MultiHeadNet_classifcation_portion             = net_structure_settings['ResNet_MultiHeadNet']['classifcation_portion']

LateFusion_setting      = net_structure_settings['LateFusion']['Settings']
LateFusion_Blocks       = net_structure_settings['LateFusion']['Blocks']
LateFusion_blocks_num   = net_structure_settings['LateFusion']['blocks_num']
LateFusion_initial_channels     = net_structure_settings['LateFusion']['initial_channels']
LateFusion_latefusion_channels  = net_structure_settings['LateFusion']['LateFusion_channels']

MultiHeadLateFusion_settings               = net_structure_settings['MultiHeadLateFusion']['Settings']
MultiHeadLateFusion_Blocks                 = net_structure_settings['MultiHeadLateFusion']['Blocks']
MultiHeadLateFusion_blocks_num             = net_structure_settings['MultiHeadLateFusion']['blocks_num']
MultiHeadLateFusion_initial_channels       = net_structure_settings['MultiHeadLateFusion']['initial_channels']
MultiHeadLateFusion_LateFusion_channels    = net_structure_settings['MultiHeadLateFusion']['LateFusion_channels']
MultiHeadLateFusion_left_bin               = net_structure_settings['MultiHeadLateFusion']['left_bin']
MultiHeadLateFusion_right_bin              = net_structure_settings['MultiHeadLateFusion']['right_bin']
MultiHeadLateFusion_bins_number            = net_structure_settings['MultiHeadLateFusion']['bins_number']
MultiHeadLateFusion_regression_portion     = net_structure_settings['MultiHeadLateFusion']['regression_portion']
MultiHeadLateFusion_classifcation_portion  = net_structure_settings['MultiHeadLateFusion']['classifcation_portion']
#######################################################################################
# Optimizer settings

Optimizer_settings = cfg['Training-Settings']['optimizer']

Adam_settings      = Optimizer_settings['Adam']['Settings']
Adam_beta0         = Optimizer_settings['Adam']['beta0']
Adam_beta1         = Optimizer_settings['Adam']['beta1']
Adam_eps           = Optimizer_settings['Adam']['eps']

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

CosineAnnealingRestartsLR = lr_settings['CosineAnnealingRestartsLR']['Settings']
CosineAnnealingRestartsLR_T_0 = lr_settings['CosineAnnealingRestartsLR']['T_0']
CosineAnnealingRestartsLR_T_mult = lr_settings['CosineAnnealingRestartsLR']['T_mult']
CosineAnnealingRestartsLR_eta_min = lr_settings['CosineAnnealingRestartsLR']['eta_min']

#######################################################################################
# activation func settings
activation_func_settings = cfg['Training_Settings']['activation_func']
activation_func_name = activation_func_settings['activation_func_name']
ReLU_ACF = activation_func_settings['ReLU']['Settings']
Tanh_ACF = activation_func_settings['Tanh']['Settings']
GeLU_ACF = activation_func_settings['GeLU']['Settings']
Sigmoid_ACF = activation_func_settings['Sigmoid']['Settings']

#######################################################################################
# Learning Objectives Settings
learning_objective = cfg['Training-Settings']['learning-objective']

species = learning_objective['species']

bias = learning_objective['bias']
normalize_bias = learning_objective['normalize_bias']
normalize_species = learning_objective['normalize_species']
absolute_species = learning_objective['absolute_species']
log_species = learning_objective['log_species']

#######################################################################################
# Loss Function Settings
Loss_Func = cfg['Training-Settings']['Loss-Functions']

Loss_type = Loss_Func['Loss_type']
Classification_loss_type = Loss_Func['Classification_loss_type']
GeoMSE_Lamba1_Penalty1 = Loss_Func['GeoMSE_Lamba1_Penalty1']
GeoMSE_Lamba2_Penalty2 = Loss_Func['GeoMSE_Lamba2_Penalty2']
GeoMSE_Gamma = Loss_Func['GeoMSE_Gamma']
Adj_GeoMSE_alpha = Loss_Func['Adj_GeoMSE_alpha']
Adj_GeoMSE_beta = Loss_Func['Adj_GeoMSE_beta']
ResNet_MultiHeadNet_regression_loss_coefficient = Loss_Func['ResNet_MultiHeadNet_regression_loss_coefficient']
ResNet_MultiHeadNet_classfication_loss_coefficient = Loss_Func['ResNet_MultiHeadNet_classfication_loss_coefficient']
#######################################################################################
# CombineWithGeophysical Settings
CombineWithGeophysical = cfg['Training-Settings']['CombineWithGeophysical']

combine_with_GeophysicalSpeceis_Switch                = CombineWithGeophysical['combine_with_GeophysicalSpeceis_Switch']
cutoff_size                                           = CombineWithGeophysical['cutoff_size']



training_infile = training_infile.format(species,species)
geophysical_biases_data_infile  = geophysical_biases_data_infile.format(species)
geophysical_species_data_infile = geophysical_species_data_infile.format(species)
ground_observation_data_infile  = ground_observation_data_infile.format(species)

def activation_function_table():
    if ReLU_ACF == True:
        return 'relu' #nn.ReLU()
    elif Tanh_ACF == True:
        return 'tanh' #nn.Tanh()
    elif GeLU_ACF == True:
        return 'gelu' #nn.GELU()
    elif Sigmoid_ACF == True:
        return 'sigmoid' #nn.Sigmoid()
    

def lr_strategy_lookup_table(optimizer):
    if ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=ExponentialLR_gamma)
    elif CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=CosineAnnealingLR_T_max,eta_min=CosineAnnealingLR_eta_min)
    elif CosineAnnealingRestartsLR:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=CosineAnnealingRestartsLR_T_0,T_mult=CosineAnnealingRestartsLR_T_mult,eta_min=CosineAnnealingLR_eta_min)



def find_latfusion_index(total_channel_names,initial_channels,late_fusion_channels):
    initial_channel_index = []
    for i in range(len(initial_channels)):
        initial_channel_index.append(total_channel_names.index(initial_channels[i]))
    
    latefusion_channel_index = []
    for i in range(len(late_fusion_channels)):
        latefusion_channel_index.append(total_channel_names.index(late_fusion_channels[i]))
    
    return initial_channel_index, latefusion_channel_index
    

def optimizer_lookup(model_parameters,learning_rate):
    if Adam_settings:
        return torch.optim.Adam(params=model_parameters,betas=(Adam_beta0, Adam_beta1),eps=Adam_eps, lr=learning_rate)
    
def Get_channel_names(channels_to_exclude:list):
    if ResNet_setting or ResNet_MLP_setting or ResNet_Classification_Settings or ResNet_MultiHeadNet_Settings:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
    elif LateFusion_setting:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_latefusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_latefusion_channels.copy()
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in side_channel_names:
                    side_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the side channel list.'.format(channels_to_exclude[ichannel]))
    elif MultiHeadLateFusion_settings:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in side_channel_names:
                    side_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the side channel list.'.format(channels_to_exclude[ichannel]))

    return total_channel_names, main_stream_channel_names, side_channel_names