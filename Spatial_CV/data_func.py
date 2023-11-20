import torch
import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
from Spatial_CV.Model_Func import predict, train, weight_reset
from Spatial_CV.Statistic_Func import linear_regression, regress2, Cal_RMSE, Calculate_PWA_PM25
from Spatial_CV.Net_Construction import  ResNet, BasicBlock, Bottleneck, Net
from Spatial_CV.visualization import regression_plot, bias_regression_plot,PM25_histgram_distribution_plot,regression_plot_area_test_average,PM25_histgram_distribution_area_tests_plot,regression_plot_ReducedAxisReduced
from Spatial_CV.ConvNet_Data import normalize_Func, Normlize_Training_Datasets, Normlize_Testing_Datasets, Data_Augmentation, Get_GeophysicalPM25_Datasets
from Spatial_CV.utils import *
from .Model_Func import MyLoss,initialize_weights_kaiming,weight_init_normal
import random
import csv


def initialize_AVD_DataRecording(Areas:list,Area_beginyears:dict,endyear:int):
    """This is used to return data recording dict. dict = { area: {Year : {Month : np.array() }}}

    Args:
        Areas (list): _description_
        Area_beginyears (dict): _description_
        endyear (int): _description_

    Returns:
        _type_: _description_
    """
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    final_data_recording = {}
    obs_data_recording = {}
    geo_data_recording = {}
    testing_population_data_recording  = {}
    training_final_data_recording = {}
    training_obs_data_recording = {}
    training_dataForSlope_recording = {}

    for iarea in Areas:
        final_data_recording[iarea] = {}
        obs_data_recording[iarea] = {}
        geo_data_recording[iarea] = {}
        testing_population_data_recording[iarea] = {}
        training_final_data_recording[iarea] = {}
        training_obs_data_recording[iarea] = {}
        training_dataForSlope_recording[iarea] = {}
        
        for iyear in range(endyear-Area_beginyears[iarea]+1):    
            final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            training_dataForSlope_recording[iarea][str(Area_beginyears[iarea]+iyear)] = {}

            for imonth in MONTH:
                final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = np.array([],dtype=np.float64)
                obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = np.array([],dtype=np.float64)
                geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = np.array([],dtype=np.float64)
                testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = np.array([],dtype=np.float64)
                training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = np.array([],dtype=np.float64)
                training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = np.array([],dtype=np.float64)
                training_dataForSlope_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = np.array([],dtype=np.float64)
             
    return final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording

def initialize_AVD_CV_dict(Areas:list,Area_beginyears:dict,endyear:int):

    return