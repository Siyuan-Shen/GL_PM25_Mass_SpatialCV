import torch
import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv
import shap

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.Model_Func import train, predict
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_sart_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Get_month_based_XIndex,Get_month_based_YIndex,Get_month_based_XY_indices,GetXIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import *
from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure, SHAPvalues_Analysis_figure


def Spatial_CV_SHAP_Analysis(width, height, sitesnumber,start_YYYY, TrainingDatasets, total_channel_names,main_stream_channel_names, side_stream_nchannel_names,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(total_channel_names)
    seed       = 19960130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    if SHAP_Analysis_Calculation_Switch:
        site_index = np.array(range(sitesnumber))
        Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
        rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
        count = 0
        shap_values_values, shap_values_base,shap_values_data = np.zeros([0,nchannel,width,height],dtype=np.float32),np.array([],dtype=np.float32),np.zeros([0,nchannel,width,height],dtype=np.float32) #initialize_AVD_SHAPValues_DataRecording(beginyear=test_beginyear,endyear=test_endyear)
        for train_index, test_index in rkf.split(site_index):
            for imodel_year in range(len(beginyears)):
                Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                for iyear in range((endyears[imodel_year]-beginyears[imodel_year]+1)):          
                    for imodel_month in range(len(training_months)):
                        
                        cnn_model = load_month_based_model(model_indir=model_outdir,typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],
                                                            month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, 
                                                        special_name=special_name, count=count, width=width, height=height)
                        cnn_model.eval()
                        yearly_test_index   = Get_month_based_XIndex(index=test_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_index  = Get_month_based_XIndex(index=train_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_Yindex  = Get_month_based_YIndex(index=test_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_Yindex = Get_month_based_YIndex(index=train_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_input   = Normalized_TrainingData[yearly_test_index,:,:,:]
                        yearly_train_input  = Normalized_TrainingData[yearly_train_index,:,:,:]
                        background_data_number = min(len(yearly_train_index),SHAP_Analysis_background_number)
                        data_to_explain_number = min(len(yearly_test_index), SHAP_Analysis_test_number)
                        Back_Ground_Data = torch.Tensor(yearly_train_input[np.sort(np.random.choice(yearly_train_input.shape[0],background_data_number, replace=False))])
                        Data_to_Explain  = torch.Tensor(yearly_test_input[np.sort(np.random.choice(yearly_test_input.shape[0], data_to_explain_number, replace=False))])
                        print('Data_to_Explain.shape: {}, type: {}'.format(Data_to_Explain.shape, type(Data_to_Explain)))
                        Back_Ground_Data = Back_Ground_Data.to(device)
                        Data_to_Explain  = Data_to_Explain.to(device)
                        CNNModel_Explainer = shap.DeepExplainer(model=cnn_model,data=Back_Ground_Data)
                        #CNNModel_Explainer =  shap.Explainer(model=cnn_model,data=Back_Ground_Data)
                        shap_values = CNNModel_Explainer.shap_values(Data_to_Explain,check_additivity=False)
                        shap_values = np.squeeze(shap_values)
                        print(shap_values.shape)
                        Data_to_Explain = Data_to_Explain.cpu().detach().numpy()
                        shap_values_values = np.append(shap_values_values, shap_values, axis=0)
                        shap_values_data   = np.append(shap_values_data, Data_to_Explain, axis=0)

        save_SHAPValues_data_recording(shap_values_values=shap_values_values, shap_values_data=shap_values_data,
                                    species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,
                                    width=width,height=height)
    if SHAP_Analysis_visualization_Switch:
        shap_values_values, shap_values_data = load_SHAPValues_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,
                                                                    width=width,height=height)
        if SHAP_Analysis_plot_type == 'beeswarm':
            shap_values_values = np.sum(shap_values_values, axis=(2,3))
            shap_values_data   = np.sum(shap_values_data, axis=(2,3))
            shap_values_with_feature_names = shap.Explanation(values=shap_values_values,data=shap_values_data,feature_names=total_channel_names)
        SHAPvalues_Analysis_figure(shap_values_with_feature_names=shap_values_with_feature_names,plot_type=SHAP_Analysis_plot_type,typeName=typeName,
                                   species=species,version=version,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,width=width,height=height,special_name=special_name)
    return

