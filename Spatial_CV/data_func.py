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


def initialize_AVD_DataRecording(Areas:list,beginyear:int,endyear:int):
    """This is used to return data recording dict. dict = { area: {Year : {Month : np.array() }}}

    Args:
        Areas (list): _description_
        Area_beginyears (dict): _description_
        endyear (int): _description_

    Returns:
        _type_: _description_
    """
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    
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
        
        for iyear in range(endyear-beginyear+1): 
            print(str(beginyear+iyear))   
            final_data_recording[iarea][str(beginyear+iyear)] = {}
            obs_data_recording[iarea][str(beginyear+iyear)] = {}
            geo_data_recording[iarea][str(beginyear+iyear)] = {}
            testing_population_data_recording[iarea][str(beginyear+iyear)] = {}
            training_final_data_recording[iarea][str(beginyear+iyear)] = {}
            training_obs_data_recording[iarea][str(beginyear+iyear)] = {}
            training_dataForSlope_recording[iarea][str(beginyear+iyear)] = {}

            for imonth in MONTH:
                final_data_recording[iarea][str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                obs_data_recording[iarea][str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                geo_data_recording[iarea][str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                testing_population_data_recording[iarea][str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_final_data_recording[iarea][str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_obs_data_recording[iarea][str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_dataForSlope_recording[iarea][str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
             
    return final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording

def initialize_AVD_CV_dict(Areas:list,Area_beginyears:dict,endyear:int):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    test_CV_R2   = {}
    train_CV_R2  = {}
    geo_CV_R2    = {}
    RMSE_CV_R2   = {}
    slope_CV_R2  = {}
    PWAModel     = {}
    PWAMonitors  = {}
    for iarea in Areas:
        test_CV_R2[iarea]  = {}
        train_CV_R2[iarea] = {}
        geo_CV_R2[iarea]   = {}
        RMSE_CV_R2[iarea]  = {}
        slope_CV_R2[iarea] = {}
        PWAModel[iarea]    = {} 
        PWAMonitors[iarea] = {}
        for iyear in range(endyear-Area_beginyears[iarea]+1):
            test_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]  = {}
            train_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            geo_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]   = {}
            RMSE_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]  = {}
            slope_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            PWAModel[iarea][str(Area_beginyears[iarea]+iyear)]    = {}
            PWAMonitors[iarea][str(Area_beginyears[iarea]+iyear)] = {}
            
            for imonth in MONTH:
                test_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth]  = -1.0
                train_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = -1.0
                geo_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth]   = -1.0
                RMSE_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth]  = -1.0
                slope_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = -1.0
                PWAModel[iarea][str(Area_beginyears[iarea]+iyear)][imonth]    = -1.0
                PWAMonitors[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = -1.0

    return test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, slope_CV_R2, slope_CV_R2, PWAModel, PWAMonitors


def initialize_AVD_CV_Alltime_dict(Areas:list,Area_beginyears:dict,endyear:int):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    test_CV_R2_Alltime   = {}
    train_CV_R2_Alltime  = {}
    geo_CV_R2_Alltime    = {}
    RMSE_CV_R2_Alltime   = {}
    slope_CV_R2_Alltime  = {}
    PWAModel_Alltime     = {}
    PWAMonitors_Alltime  = {}
    for iarea in Areas:
        test_CV_R2_Alltime[iarea]  = {'Alltime':{}}
        train_CV_R2_Alltime[iarea] = {'Alltime':{}}
        geo_CV_R2_Alltime[iarea]   = {'Alltime':{}}
        RMSE_CV_R2_Alltime[iarea]  = {'Alltime':{}}
        slope_CV_R2_Alltime[iarea] = {'Alltime':{}}
        PWAModel_Alltime[iarea]    = {'Alltime':{}}
        PWAMonitors_Alltime[iarea] = {'Alltime':{}}
        for imonth in MONTH:
            ## np.zeros((3),dtype=np.float64) - 0 - mean, 1 - min, 2 - max
            test_CV_R2_Alltime[iarea]['Alltime'][imonth]  = np.zeros((3),dtype=np.float64)
            train_CV_R2_Alltime[iarea]['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
            geo_CV_R2_Alltime[iarea]['Alltime'][imonth]   = np.zeros((3),dtype=np.float64)
            RMSE_CV_R2_Alltime[iarea]['Alltime'][imonth]  = np.zeros((3),dtype=np.float64)
            slope_CV_R2_Alltime[iarea]['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
            PWAModel_Alltime[iarea]['Alltime'][imonth]    = np.zeros((3),dtype=np.float64)
            PWAMonitors_Alltime[iarea]['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
    return test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_CV_R2_Alltime, slope_CV_R2_Alltime, PWAModel_Alltime, PWAMonitors_Alltime

def calculate_Statistics_results(Areas:list,Area_beginyears:dict,endyear:int,final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, slope_CV_R2, slope_CV_R2, PWAModel, PWAMonitors = initialize_AVD_CV_dict(Areas=Areas,Area_beginyears=Area_beginyears,endyear=endyears[-1])
    for iarea in Areas:
        for iyear in range(endyear-Area_beginyears[iarea]+1):
            for imonth in MONTH:
                print('Area: {}, Year: {}, Month: {}'.format(iarea, Area_beginyears[iarea]+iyear, imonth))
                test_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = linear_regression(final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth], obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                train_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = linear_regression(training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth], training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                geo_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = linear_regression(geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth], obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                RMSE_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = Cal_RMSE(final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth], obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                regression_Dic = regress2(_x= obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth],_y=final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
                slope_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = slope
                PWAModel[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = Calculate_PWA_PM25(Population_array=testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth],PM25_array=final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                PWAMonitors[iarea][str(Area_beginyears[iarea]+iyear)][imonth] = Calculate_PWA_PM25(Population_array=testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth],PM25_array=obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth])

                if imonth == 'Jan':
                    final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                else:
                    final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] += final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] += obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] += geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] += testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] += training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] += training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)][imonth]
                    
            final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual']/12.0
            obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual']/12.0
            geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual']/12.0
            testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual']/12.0
            training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual']/12.0
            training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual']/12.0
            
            print('Area: {}, Year: {}, Month: {}'.format(iarea, Area_beginyears[iarea]+iyear, 'Annual'))
            test_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = linear_regression(final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'], obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'])
            train_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = linear_regression(training_final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'], training_obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'])
            geo_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = linear_regression(geo_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'], obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'])
            RMSE_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = Cal_RMSE(final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'], obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'])
            regression_Dic = regress2(_x= obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'],_y=final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
            slope_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = slope
            PWAModel[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = Calculate_PWA_PM25(Population_array=testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'],PM25_array=final_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'])
            PWAMonitors[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'] = Calculate_PWA_PM25(Population_array=testing_population_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'],PM25_array=obs_data_recording[iarea][str(Area_beginyears[iarea]+iyear)]['Annual'])


    return test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, slope_CV_R2, PWAModel, PWAMonitors


def calculate_Alltime_Statistics_results(Areas:list,Area_beginyears:dict,endyear:int,test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, slope_CV_R2, PWAModel, PWAMonitors):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','Annual']
    test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_CV_R2_Alltime, slope_CV_R2_Alltime, PWAModel_Alltime, PWAMonitors_Alltime = initialize_AVD_CV_Alltime_dict(Areas=Areas,Area_beginyears=Area_beginyears,endyear=endyear)
    for iarea in Areas:
        for imonth in MONTH:
            temp_test_CV_R2_Alltime   = np.array([],dtype=np.float64)
            temp_train_CV_R2_Alltime  = np.array([],dtype=np.float64)
            temp_geo_CV_R2_Alltime    = np.array([],dtype=np.float64)
            temp_RMSE_CV_R2_Alltime   = np.array([],dtype=np.float64)
            temp_slope_CV_R2_Alltime  = np.array([],dtype=np.float64)
            temp_PWAModel_Alltime     = np.array([],dtype=np.float64)
            temp_PWAMonitors_Alltime  = np.array([],dtype=np.float64)
            for iyear in range(endyear-Area_beginyears[iarea]+1):
                print('Area: {}, Year: {}, Month: {}'.format(iarea, Area_beginyears[iarea]+iyear, imonth))
                temp_test_CV_R2_Alltime  = np.append(temp_test_CV_R2_Alltime, test_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                temp_train_CV_R2_Alltime = np.append(temp_train_CV_R2_Alltime, train_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                temp_geo_CV_R2_Alltime   = np.append(temp_geo_CV_R2_Alltime, geo_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                temp_RMSE_CV_R2_Alltime  = np.append(temp_RMSE_CV_R2_Alltime, RMSE_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                temp_slope_CV_R2_Alltime = np.append(temp_slope_CV_R2_Alltime, slope_CV_R2[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                temp_PWAModel_Alltime    = np.append(temp_PWAModel_Alltime, PWAModel[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
                temp_PWAMonitors_Alltime = np.append(temp_PWAMonitors_Alltime, PWAMonitors[iarea][str(Area_beginyears[iarea]+iyear)][imonth])
            
            test_CV_R2_Alltime[iarea]['Alltime'][imonth]     = get_mean_min_max_statistic(temp_test_CV_R2_Alltime)
            train_CV_R2_Alltime[iarea]['Alltime'][imonth]    = get_mean_min_max_statistic(temp_train_CV_R2_Alltime)
            geo_CV_R2_Alltime[iarea]['Alltime'][imonth]      = get_mean_min_max_statistic(temp_geo_CV_R2_Alltime)
            RMSE_CV_R2_Alltime[iarea]['Alltime'][imonth]     = get_mean_min_max_statistic(temp_RMSE_CV_R2_Alltime)
            slope_CV_R2_Alltime[iarea]['Alltime'][imonth]    = get_mean_min_max_statistic(temp_slope_CV_R2_Alltime)
            PWAModel_Alltime[iarea]['Alltime'][imonth]       = get_mean_min_max_statistic(temp_PWAModel_Alltime)
            PWAMonitors_Alltime[iarea]['Alltime'][imonth]    = get_mean_min_max_statistic(temp_PWAMonitors_Alltime)

    return test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_CV_R2_Alltime, slope_CV_R2_Alltime, PWAModel_Alltime, PWAMonitors_Alltime

def get_longterm_array(area, imonth, beginyear, endyear, final_data_recording,obs_data_recording):

    final_longterm_data = np.zeros(final_data_recording[area][str(beginyear)]['Jan'].shape, dtype=np.float64)
    obs_longterm_data   = np.zeros(final_data_recording[area][str(beginyear)]['Jan'].shape, dtype=np.float64)
    if imonth == 'Annual':
        count = 0
        MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for iyear in range(endyear-beginyear+1):
            for imonth in range(len(MONTH)):
                final_longterm_data += final_data_recording[str(beginyear+iyear)][MONTH[imonth]]
                obs_longterm_data   += obs_data_recording[str(beginyear+iyear)][MONTH[imonth]]
                count += 1
        final_longterm_data = final_longterm_data/count
        obs_longterm_data   = obs_longterm_data/count
    else:
        for iyear in range(endyear-beginyear+1):
            final_longterm_data += final_data_recording[area][str(beginyear+iyear)][imonth]
            obs_longterm_data   += obs_data_recording[area][str(beginyear+iyear)][imonth]
        final_longterm_data = final_longterm_data/(endyear-beginyear+1.0)
        obs_longterm_data   = obs_longterm_data/(endyear-beginyear+1.0)
    return final_longterm_data, obs_longterm_data

def get_mean_min_max_statistic(temp_CV):
    temp_array = np.zeros((3),dtype=np.float64)
    temp_array[0] = np.mean(temp_CV)
    temp_array[1] = np.min(temp_CV)
    temp_array[2] = np.max(temp_CV)
    return temp_array