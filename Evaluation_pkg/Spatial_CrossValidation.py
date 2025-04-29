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
from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure


def AVD_Spatial_CrossValidation(width, height, sitesnumber,start_YYYY, TrainingDatasets, total_channel_names,main_stream_channel_names, side_stream_nchannel_names):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
    population_data = load_coMonitor_Population()
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(total_channel_names)
    seed       = 19980130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(beginyear=beginyears[0],endyear=endyears[-1])
    Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording = initialize_Loss_Accuracy_Recordings(kfolds=kfold,n_models=len(beginyears)*len(training_months),epoch=epoch,batchsize=batchsize)
    lat_test_recording = np.array([],dtype=np.float32)
    lon_test_recording = np.array([],dtype=np.float32)

    count = 0
    if not Spatial_CV_test_only_Switch:
        for train_index, test_index in rkf.split(site_index):
            lat_test_recording = np.append(lat_test_recording,lat[test_index])
            lon_test_recording = np.append(lon_test_recording,lon[test_index])
            for imodel_year in range(len(beginyears)):
                Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                for imodel_month in range(len(training_months)):
                
                    X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_month_based_XY_indices(train_index=train_index,test_index=test_index,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_index=training_months[imodel_month], sitesnumber=sitesnumber)
                    X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)
                    #print('X_train size: {}, X_test size: {}, y_train size: {}, y_test size: {} -------------------------------------------'.format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
                    # *------------------------------------------------------------------------------*#
                    ## Training Process.
                    # *------------------------------------------------------------------------------*#

                    cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_nchannel_names))

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    cnn_model.to(device)
                    torch.manual_seed(21)
                    train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, input_std=input_std,input_mean=input_mean,width=width,height=height,BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch,
                                                                    initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_nchannel_names)
                    Training_losses_recording[count,imodel_year*len(training_months)+imodel_month,0:len(train_loss)] = train_loss
                    Training_acc_recording[count,imodel_year*len(training_months)+imodel_month,0:len(train_acc)]    = train_acc
                    valid_losses_recording[count,imodel_year*len(training_months)+imodel_month,0:len(valid_losses)]  = valid_losses
                    valid_acc_recording[count,imodel_year*len(training_months)+imodel_month,0:len(test_acc)]       = test_acc

                    save_trained_month_based_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height)
                for iyear in range((endyears[imodel_year]-beginyears[imodel_year]+1)):
                    for imodel_month in range(len(training_months)):
                        yearly_test_index   = Get_month_based_XIndex(index=test_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_index  = Get_month_based_XIndex(index=train_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_Yindex  = Get_month_based_YIndex(index=test_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_Yindex = Get_month_based_YIndex(index=train_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_input  = Normalized_TrainingData[yearly_test_index,:,:,:]
                        yearly_train_input = Normalized_TrainingData[yearly_train_index,:,:,:]
                        
                        cnn_model = load_month_based_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height)
                        Validation_Prediction = predict(inputarray=yearly_test_input, model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_nchannel_names)
                        Training_Prediction   = predict(inputarray=yearly_train_input,  model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_nchannel_names)
                        final_data = Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,yearly_test_Yindex)
                        train_final_data = Get_final_output(Training_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,yearly_train_Yindex)
                    
                        if combine_with_GeophysicalSpeceis_Switch:
                            nearest_distance = get_nearest_test_distance(area_test_index=test_index,area_train_index=train_index,site_lat=lat,site_lon=lon)
                            coeficient = get_coefficients(nearest_site_distance=nearest_distance,cutoff_size=cutoff_size,beginyear=beginyears[imodel_year],
                                                endyear = endyears[imodel_year],months=training_months[imodel_month])
                            final_data = (1.0-coeficient)*final_data + coeficient * geophysical_species[yearly_test_Yindex]
                        if ForcedSlopeUnity:
                            final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=SPECIES_OBS[yearly_train_Yindex]
                                                    ,test_final_data=final_data,train_area_index=train_index,test_area_index=test_index,
                                                    endyear=beginyears[imodel_year]+iyear,beginyear=beginyears[imodel_year]+iyear,month_index=training_months[imodel_month],EachMonth=EachMonthForcedSlopeUnity)

                        # *------------------------------------------------------------------------------*#
                        ## Recording observation and prediction for this model this fold.
                        # *------------------------------------------------------------------------------*#

                        Validation_obs_data   = SPECIES_OBS[yearly_test_Yindex]
                        Training_obs_data     = SPECIES_OBS[yearly_train_Yindex]
                        Geophysical_test_data = geophysical_species[yearly_test_Yindex]
                        population_test_data  = population_data[yearly_test_Yindex]

                        for imonth in range(len(training_months[imodel_month])):
                            final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]              = np.append(final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], final_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Validation_obs_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Geophysical_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]     = np.append(training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], train_final_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                            training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]       = np.append(training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Training_obs_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                            testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]] = np.append(testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], population_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                

            count += 1
        save_month_based_data_recording(obs_data=obs_data_recording,final_data=final_data_recording,geo_data_recording=geo_data_recording,training_final_data_recording=training_final_data_recording,
                                        training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,lat_recording=lat_test_recording,lon_recording=lon_test_recording,
                                        species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height)
    
        save_loss_accuracy(model_outdir=model_outdir,loss=Training_losses_recording, accuracy=Training_acc_recording,valid_loss=valid_losses_recording, valid_accuracy=valid_acc_recording,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height)
    txtfile_outdir = txt_outdir + '{}/{}/Results/results-SpatialCV/statistical_indicators/{}_{}_{}_{}Channel_{}x{}{}/'.format(species, version,typeName,species,version,nchannel,width,height,special_name)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    obs_data_recording, final_data_recording, geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_test_recording, lon_test_recording = load_month_based_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height)
    for iyear in range(len(test_beginyears)):
        test_beginyear = test_beginyears[iyear]
        test_endyear   = test_endyears[iyear]
        test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors, regional_number = calculate_Statistics_results(test_beginyear=test_beginyear, test_endyear=test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=site_index,Area='Global')
        txt_outfile =  txtfile_outdir + 'AVDSpatialCV_{}-{}_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(test_beginyear,test_endyear,typeName,species,version,nchannel,width,height,special_name)
        AVD_output_text(outfile=txt_outfile,status='w', Area='Global',test_beginyears=test_beginyear,test_endyears=test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                            slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)
    for iregion in additional_test_regions:
        mask_map, mask_lat, mask_lon = load_GL_extent_Mask(region_name=iregion)
        masked_array_index = find_masked_latlon(mask_map=mask_map,mask_lat=mask_lat,mask_lon=mask_lon,test_lat=lat_test_recording,test_lon=lon_test_recording)
        for iyear in range(len(test_beginyears)):
            test_beginyear = test_beginyears[iyear]
            test_endyear   = test_endyears[iyear]
            test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors,regional_number = calculate_Statistics_results(test_beginyear=test_beginyear, test_endyear=test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=masked_array_index,Area=iregion)
            txt_outfile =  txtfile_outdir + 'AVDSpatialCV_{}-{}_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(test_beginyear,test_endyear,typeName,species,version,nchannel,width,height,special_name)
            AVD_output_text(outfile=txt_outfile,status='a', Area=iregion,test_beginyears=test_beginyear,test_endyears=test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                                slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)
    
    
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=test_beginyear, endyear=test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
    save_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=test_beginyear, imonth=imonth,endyear=test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
        save_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height)
    return


def FixedNumber_AVD_Spatial_CrossValidation(Fixednumber_test_site,Fixednumber_train_site,width, height, sitesnumber,start_YYYY, TrainingDatasets,total_channel_names,main_stream_channel_names, side_stream_nchannel_names):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    beginyears = Fixnumber_beginyears
    endyears   = Fixnumber_test_endyear
    training_months = Fixnumber_training_months
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
    population_data = load_coMonitor_Population()
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(total_channel_names)
    seed       = 19980130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=Fixnumber_kfold, n_repeats=Fixnumber_repeats, random_state=seed)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(beginyear=beginyears[0],endyear=endyears[-1])
    Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording = initialize_Loss_Accuracy_Recordings(kfolds=Fixnumber_kfold,n_models=len(beginyears)*len(training_months),epoch=epoch,batchsize=batchsize)
    lat_test_recording = np.array([],dtype=np.float32)
    lon_test_recording = np.array([],dtype=np.float32)
    count = 0
    if not Fixnumber_Spatial_CV_test_only_Switch:
        for init_train_index, init_test_index in rkf.split(site_index):
            train_index, test_index = GetFixedNumber_TrainingIndex(test_index=init_test_index,train_index=init_train_index,fixed_test_number=Fixednumber_test_site,fixed_train_number=Fixednumber_train_site)
            lat_test_recording = np.append(lat_test_recording,lat[test_index])
            lon_test_recording = np.append(lon_test_recording,lon[test_index])
            for imodel_year in range(len(beginyears)):
                Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                for imodel_month in range(len(training_months)):
                    X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_month_based_XY_indices(train_index=train_index,test_index=test_index,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_index=training_months[imodel_month], sitesnumber=sitesnumber)
                    X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)
                    #print('X_train size: {}, X_test size: {}, y_train size: {}, y_test size: {} -------------------------------------------'.format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
                    # *------------------------------------------------------------------------------*#
                    ## Training Process.
                    # *------------------------------------------------------------------------------*#

                    cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_nchannel_names))

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    cnn_model.to(device)
                    torch.manual_seed(21)
                    train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, input_std=input_std,input_mean=input_mean,width=width,height=height,BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch,
                                                                    initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_nchannel_names)
                    Training_losses_recording[count,imodel_year*len(training_months)+imodel_month,0:len(train_loss)] = train_loss
                    Training_acc_recording[count,imodel_year*len(training_months)+imodel_month,0:len(train_acc)]    = train_acc
                    valid_losses_recording[count,imodel_year*len(training_months)+imodel_month,0:len(valid_losses)]  = valid_losses
                    valid_acc_recording[count,imodel_year*len(training_months)+imodel_month,0:len(test_acc)]       = test_acc


                    save_trained_month_based_FixNumber_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height,fixed_test_number=Fixednumber_test_site,fixed_train_number=Fixednumber_train_site)
                for iyear in range((endyears[imodel_year]-beginyears[imodel_year]+1)):
                    for imodel_month in range(len(training_months)):
                        yearly_test_index   = Get_month_based_XIndex(index=test_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_index  = Get_month_based_XIndex(index=train_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_Yindex  = Get_month_based_YIndex(index=test_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_Yindex = Get_month_based_YIndex(index=train_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_input  = Normalized_TrainingData[yearly_test_index,:,:,:]
                        yearly_train_input = Normalized_TrainingData[yearly_train_index,:,:,:]
                        
                        cnn_model = load_trained_month_based_FixNumber_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height,fixed_test_number=Fixednumber_test_site,fixed_train_number=Fixednumber_train_site)
                        Validation_Prediction = predict(inputarray=yearly_test_input, model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_nchannel_names)
                        Training_Prediction   = predict(inputarray=yearly_train_input,  model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_nchannel_names)
                        final_data = Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,yearly_test_Yindex)
                        train_final_data = Get_final_output(Training_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,yearly_train_Yindex)
                    
                        if combine_with_GeophysicalSpeceis_Switch:
                            nearest_distance = get_nearest_test_distance(area_test_index=test_index,area_train_index=train_index,site_lat=lat,site_lon=lon)
                            coeficient = get_coefficients(nearest_site_distance=nearest_distance,cutoff_size=cutoff_size,beginyear=beginyears[imodel_year],
                                                endyear = endyears[imodel_year])
                            final_data = (1.0-coeficient)*final_data + coeficient * geophysical_species[yearly_test_Yindex]
                        if ForcedSlopeUnity:
                            final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=SPECIES_OBS[yearly_train_Yindex]
                                                    ,test_final_data=final_data,train_area_index=train_index,test_area_index=test_index,
                                                    endyear=beginyears[imodel_year]+iyear,beginyear=beginyears[imodel_year]+iyear,month_index=training_months[imodel_month],EachMonth=EachMonthForcedSlopeUnity)

                        # *------------------------------------------------------------------------------*#
                        ## Recording observation and prediction for this model this fold.
                        # *------------------------------------------------------------------------------*#

                        Validation_obs_data   = SPECIES_OBS[yearly_test_Yindex]
                        Training_obs_data     = SPECIES_OBS[yearly_train_Yindex]
                        Geophysical_test_data = geophysical_species[yearly_test_Yindex]
                        population_test_data  = population_data[yearly_test_Yindex]

                        for imonth in range(len(training_months[imodel_month])):
                            final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]              = np.append(final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], final_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Validation_obs_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Geophysical_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]     = np.append(training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], train_final_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                            training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]       = np.append(training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Training_obs_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                            testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]] = np.append(testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], population_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                

            count += 1
        save_Fixnumber_month_based_data_recording(obs_data=obs_data_recording,final_data=final_data_recording,geo_data_recording=geo_data_recording,training_final_data_recording=training_final_data_recording,
                                                  training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,lat_recording=lat_test_recording,lon_recording=lon_test_recording,
                                        species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,test_number=Fixednumber_test_site,train_number=Fixednumber_train_site)
    obs_data_recording, final_data_recording, geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_test_recording, lon_test_recording = load_Fixnumber_month_based_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,test_number=Fixednumber_test_site,train_number=Fixednumber_train_site)

    txtfile_outdir = txt_outdir + '{}/{}/Results/results-FixNumberCV/statistical_indicators/{}_{}_{}_{}Channel_{}testsites_{}trainsites_{}x{}{}/'.format(species, version,typeName,species,version,nchannel,Fixednumber_test_site,Fixednumber_train_site,width,height,special_name)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    
    for iyear in range(len(Fixnumber_test_beginyears)):
        Fixnumber_test_beginyear = Fixnumber_test_beginyears[iyear]
        Fixnumber_test_endyear   = Fixnumber_test_endyears[iyear]
        txt_outfile =  txtfile_outdir + 'AVDSpatialCV_{}-{}_{}_{}_{}_{}Channel_{}testsites_{}trainsites_{}x{}{}.csv'.format(Fixnumber_test_beginyear,Fixnumber_test_endyear,typeName,species,version,nchannel,Fixednumber_test_site,Fixednumber_train_site,width,height,special_name)
        test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors,regional_number = calculate_Statistics_results(test_beginyear=Fixnumber_test_beginyear, test_endyear=Fixnumber_test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=site_index,Area='Global')
        AVD_output_text(outfile=txt_outfile,status='w',Area='Global', test_beginyears=Fixnumber_test_beginyear,test_endyears=Fixnumber_test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                        slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)
    for iregion in Fixnumber_additional_test_regions:
        mask_map, mask_lat, mask_lon = load_GL_Mask_data(region_name=iregion)
        masked_array_index = find_masked_latlon(mask_map=mask_map,mask_lat=mask_lat,mask_lon=mask_lon,test_lat=lat_test_recording,test_lon=lon_test_recording)
        for iyear in range(len(Fixnumber_test_beginyears)):
            Fixnumber_test_beginyear = test_beginyears[iyear]
            Fixnumber_test_endyear   = test_endyears[iyear]
        
            test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors, regional_number = calculate_Statistics_results(test_beginyear=Fixnumber_test_beginyear, test_endyear=Fixnumber_test_beginyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=masked_array_index,Area=iregion)
            txt_outfile =  txtfile_outdir + 'AVDSpatialCV_{}-{}_{}_{}_{}_{}Channel_{}testsites_{}trainsites_{}x{}{}.csv'.format(Fixnumber_test_beginyear,Fixnumber_test_endyear,typeName,species,version,nchannel,Fixednumber_test_site,Fixednumber_train_site,width,height,special_name)
            AVD_output_text(outfile=txt_outfile,status='a', Area=iregion,test_beginyears=Fixnumber_test_beginyear,test_endyears=Fixnumber_test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                                slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)
    
    save_loss_accuracy(model_outdir=model_outdir,loss=Training_losses_recording, accuracy=Training_acc_recording,valid_loss=valid_losses_recording, valid_accuracy=valid_acc_recording,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height)
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=Fixnumber_test_beginyear, endyear=Fixnumber_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
    save_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=Fixnumber_test_beginyear, imonth=imonth,endyear=Fixnumber_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
        save_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height)
      



    return




def Normal_Spatial_CrossValidation(width, height, sitesnumber,start_YYYY, TrainingDatasets):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    
    nchannel   = len(channel_names)
    seed       = 19980130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)

    annual_final_data_recording, annual_obs_data_recording   = initialize_AnnualDataRecording_Dic(beginyears=beginyears)
    monthly_final_data_recording, monthly_obs_data_recording =  initialize_MonthlyDataRecording_Dic(beginyears=beginyears)
    training_monthly_final_data_recording, training_monthly_obs_data_recording =  initialize_MonthlyDataRecording_Dic(beginyears=beginyears)
    geo_monthly_final_data_recording, geo_monthly_obs_data_recording =  initialize_MonthlyDataRecording_Dic(beginyears=beginyears)
    training_CV_R2, training_month_CV_R2, training_annual_CV_R2,  geophysical_CV_R2,geophysical_annual_CV_R2,geophysical_month_CV_R2, CV_R2, CV_slope, CV_RMSE, annual_CV_R2, annual_CV_slope, annual_CV_RMSE, month_CV_R2, month_CV_slope, month_CV_RMSE = initialize_multimodels_CV_Dic(kfold=kfold,repeats=repeats,beginyears=beginyears)

    count = 0
    for train_index, test_index in rkf.split(site_index):
        Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)

        Alltime_final_test = np.array([], dtype=np.float64)
        Alltime_obs_test   = np.array([], dtype=np.float64)
        Alltime_geo_test   = np.array([], dtype=np.float64)
        Alltime_final_train = np.array([], dtype=np.float64)
        Alltime_obs_train   = np.array([], dtype=np.float64)
        
        for imodel in range(len(beginyears)):
            Normalized_TrainingData = get_trainingdata_within_sart_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel],training_end_YYYY=endyears[imodel],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
            X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_XY_indices(train_index=train_index,test_index=test_index,beginyear=beginyears[imodel],endyear=endyears[imodel], sitesnumber=sitesnumber)
            X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)

            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            if ResNet_setting:
                block = resnet_block_lookup_table(ResNet_Blocks)
                cnn_model = ResNet(nchannel=nchannel,block=block,blocks_num=ResNet_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)
            #cnn_model = Net(nchannel=nchannel)
    

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)
            train_loss, train_acc, valid_losses, test_acc = train(model=cnn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,input_std=input_std,input_mean=input_mean,width=width,height=height, BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch)
            save_trained_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height)

            # *------------------------------------------------------------------------------*#
            ## Evaluation Process.
            # *------------------------------------------------------------------------------*#
            Validation_Prediction = predict(X_test, cnn_model, 3000)
            Training_Prediction   = predict(X_train, cnn_model, 3000)
            final_data = Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,Y_Testing_index)
            train_final_data = Get_final_output(Training_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,Y_Training_index)
            if ForcedSlopeUnity:
                final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=SPECIES_OBS[Y_Training_index]
                                                   ,test_final_data=Validation_Prediction,train_area_index=train_index,test_area_index=test_index,
                                                   endyear=endyears[imodel],beginyear=beginyears[imodel],EachMonth=EachMonthForcedSlopeUnity)
            # *------------------------------------------------------------------------------*#
            ## Recording observation and prediction for this model this fold.
            # *------------------------------------------------------------------------------*#

            Validation_obs_data   = SPECIES_OBS[Y_Testing_index]
            Training_obs_data     = SPECIES_OBS[Y_Training_index]
            Geophysical_test_data = geophysical_species[Y_Testing_index]

            Alltime_final_test = np.append(Alltime_final_test, final_data)
            Alltime_obs_test   = np.append(Alltime_obs_test, Validation_obs_data)
            Alltime_final_train= np.append(Alltime_final_train, train_final_data)
            Alltime_obs_train  = np.append(Alltime_obs_train, Training_obs_data)
            Alltime_geo_test   = np.append(Alltime_geo_test, Geophysical_test_data)

            # *------------------------------------------------------------------------------*#
            ## Calculate the statistical results for this model this fold.
            # *------------------------------------------------------------------------------*#
            
            print(' fold: {},  beginyear: {}, endyear: {}'.format(count, beginyears[imodel], endyears[imodel]))
            
            ## Test fold monthly estimation
            CV_R2[str(beginyears[imodel])][count] = linear_regression(final_data, Validation_obs_data)
            CV_regression_dic = regress2(_x=Validation_obs_data, _y=final_data, _method_type_1='ordinary least square',_method_type_2='reduced major axis')
            CV_slope[str(beginyears[imodel])][count] = CV_regression_dic['slope']
            CV_RMSE[str(beginyears[imodel])][count]  = Cal_RMSE(Validation_obs_data, final_data)

            ## Test fold annual estimation
            print( 'Testing Results:')
            annual_R2,annual_final_data,annual_mean_obs,annual_slope,annual_RMSE = CalculateAnnualR2(test_index=test_index,final_data=final_data,test_obs_data=Validation_obs_data,
                                                                                       beginyear=beginyears[imodel], endyear=endyears[imodel])
            annual_final_data_recording[str(beginyears[imodel])] = np.append(annual_final_data_recording[str(beginyears[imodel])], annual_final_data)
            annual_obs_data_recording[str(beginyears[imodel])]   = np.append(annual_obs_data_recording[str(beginyears[imodel])], annual_mean_obs)
            annual_CV_R2[str(beginyears[imodel])][count]    = annual_R2
            annual_CV_slope[str(beginyears[imodel])][count] = annual_slope
            annual_CV_RMSE[str(beginyears[imodel])][count]  = annual_RMSE

            ## Test fold monthly estimation for each month
            month_R2, month_slope, month_RMSE, monthly_final_data_recording[str(beginyears[imodel])], monthly_obs_data_recording[str(beginyears[imodel])] = CalculateMonthR2(test_index=test_index, final_data = final_data, test_obs_data = Validation_obs_data,
                                                                 beginyear=beginyears[imodel], endyear=endyears[imodel], monthly_final_test_imodel=monthly_final_data_recording[str(beginyears[imodel])], monthly_obs_test_imodel=monthly_obs_data_recording[str(beginyears[imodel])])
            month_CV_R2[str(beginyears[imodel])][:,count]    = month_R2
            month_CV_slope[str(beginyears[imodel])][:,count] = month_slope
            month_CV_RMSE[str(beginyears[imodel])][:,count]  = month_RMSE

            ## Training fold 
            print( 'Training Results:')
            training_annual_R2,training_annual_final_data,training_annual_mean_obs,training_slope,training_RMSE = CalculateAnnualR2(test_index=train_index,final_data=train_final_data,test_obs_data=Training_obs_data,beginyear=beginyears[imodel], endyear=endyears[imodel])
            training_annual_CV_R2[str(beginyears[imodel])][count] = training_annual_R2 
            training_monthly_R2, training_month_slope, training_month_RMSE, training_monthly_final_data_recording[str(beginyears[imodel])], training_monthly_obs_data_recording[str(beginyears[imodel])] = CalculateMonthR2(test_index=train_index,final_data=train_final_data,test_obs_data=Training_obs_data, beginyear=beginyears[imodel], endyear=endyears[imodel],
                                                                                                                                                                                                                            monthly_final_test_imodel=training_monthly_final_data_recording[str(beginyears[imodel])], monthly_obs_test_imodel=training_monthly_obs_data_recording[str(beginyears[imodel])])
            training_month_CV_R2[str(beginyears[imodel])][:,count] = training_monthly_R2

            print( 'Geophysical Results:')
            geo_annual_R2,geo_annual_final_data,geo_annual_mean_obs,geo_slope,geo_RMSE = CalculateAnnualR2(test_index=test_index,final_data=Geophysical_test_data,test_obs_data=Validation_obs_data,beginyear=beginyears[imodel], endyear=endyears[imodel])
            geophysical_annual_CV_R2[str(beginyears[imodel])][count] = geo_annual_R2 
            geo_monthly_R2, geo_month_slope, geo_month_RMSE,geo_monthly_final_data_recording[str(beginyears[imodel])], geo_monthly_obs_data_recording[str(beginyears[imodel])] = CalculateMonthR2(test_index=test_index,final_data=Geophysical_test_data,test_obs_data=Validation_obs_data, beginyear=beginyears[imodel], endyear=endyears[imodel],
                                                                                                                                                                                                  monthly_final_test_imodel=geo_monthly_final_data_recording[str(beginyears[imodel])], monthly_obs_test_imodel=geo_monthly_obs_data_recording[str(beginyears[imodel])])
            training_month_CV_R2[str(beginyears[imodel])][:,count] = training_monthly_R2
            geophysical_month_CV_R2[str(beginyears[imodel])][:,count] = geo_monthly_R2



        print(' fold: ',str(count),  ' - Alltime')
        CV_R2['Alltime'][count] = linear_regression(Alltime_final_test, Alltime_obs_test)
        CV_regression_Dic       = regress2(_x=Alltime_obs_test, _y=Alltime_final_test,_method_type_1='ordinary least square',_method_type_2='reduced major axis')
        CV_slope['Alltime'][count] = CV_regression_Dic['slope']
        CV_RMSE['Alltime'][count]  = Cal_RMSE(Alltime_obs_test, Alltime_final_test)

        print( 'Testing Results:')
        annual_R2,annual_final_data,annual_mean_obs,annual_slope,annual_RMSE = CalculateAnnualR2(test_index=test_index, final_data=Alltime_final_test, test_obs_data=Alltime_obs_test,
                                                                                   beginyear=beginyears[0], endyear=endyears[-1])
        annual_final_data_recording['Alltime'] = np.append(annual_final_data_recording['Alltime'], annual_final_data)
        annual_obs_data_recording['Alltime']   = np.append(annual_obs_data_recording['Alltime'],   annual_mean_obs)
        annual_CV_R2['Alltime'][count]    = annual_R2
        annual_CV_slope['Alltime'][count] = annual_slope
        annual_CV_RMSE['Alltime'][count]  = annual_RMSE

        month_R2, month_slope, month_RMSE,monthly_final_data_recording['Alltime'], monthly_obs_data_recording['Alltime'] = CalculateMonthR2(test_index=test_index, final_data=Alltime_final_test, test_obs_data=Alltime_obs_test,
                                                             beginyear=beginyears[0], endyear=endyears[-1],monthly_final_test_imodel=monthly_final_data_recording['Alltime'], monthly_obs_test_imodel=monthly_obs_data_recording['Alltime'])
        month_CV_R2['Alltime'][:,count]    = month_R2
        month_CV_slope['Alltime'][:,count] = month_slope
        month_CV_RMSE['Alltime'][:,count]  = month_RMSE

        print( 'Training Results:')
        training_annual_R2, training_annual_final_data,training_annual_mean_obs,training_slope,training_RMSE = CalculateAnnualR2(test_index=train_index,final_data=Alltime_final_train,test_obs_data=Alltime_obs_train,beginyear=beginyears[0], endyear=endyears[-1])
        training_annual_CV_R2['Alltime'][count] = training_annual_R2 
        training_monthly_R2, training_month_slope, training_month_RMSE, training_monthly_final_data_recording['Alltime'], training_monthly_obs_data_recording['Alltime']  = CalculateMonthR2(test_index=train_index, final_data=Alltime_final_train,test_obs_data=Alltime_obs_train,beginyear=beginyears[0], endyear=endyears[-1],
                                                                                                                                                                                             monthly_final_test_imodel=training_monthly_final_data_recording['Alltime'], monthly_obs_test_imodel=training_monthly_obs_data_recording['Alltime'])
        training_month_CV_R2['Alltime'][:, count] = training_monthly_R2

        print( 'Geophysical Results:')
        geo_annual_R2,geo_annual_final_data,geo_annual_mean_obs,geo_slope,geo_RMSE = CalculateAnnualR2(test_index=test_index,final_data=Alltime_geo_test,test_obs_data=Alltime_obs_test,beginyear=beginyears[0], endyear=endyears[-1])
        geophysical_annual_CV_R2['Alltime'][count] = geo_annual_R2 
        geo_monthly_R2, geo_month_slope, geo_month_RMSE, geo_monthly_final_data_recording['Alltime'], geo_monthly_obs_data_recording['Alltime'] = CalculateMonthR2(test_index=test_index,final_data=Alltime_geo_test,test_obs_data=Alltime_obs_test, beginyear=beginyears[0], endyear=endyears[-1],
                                                                           monthly_final_test_imodel=geo_monthly_final_data_recording['Alltime'], monthly_obs_test_imodel=geo_monthly_obs_data_recording['Alltime'])
        geophysical_month_CV_R2['Alltime'][:,count] = geo_monthly_R2

        count += 1

    ##################################################################################################################
    # * Plot and Output
    ##################################################################################################################

    
    txtfile_outdir = txt_outdir + '{}/{}/Results/results-SpatialCV/'.format(species, version)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    
    txt_outfile =  txtfile_outdir + 'SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(typeName,species,version,nchannel,width,height,special_name)
    
    output_text(outfile=txt_outfile, status='a', CV_R2=CV_R2['Alltime'],annual_CV_R2=annual_CV_R2['Alltime'],month_CV_R2=month_CV_R2['Alltime'], 
                training_annual_CV_R2=training_annual_CV_R2['Alltime'], training_month_CV_R2=training_month_CV_R2['Alltime'],
                geo_annual_CV_R2=geophysical_annual_CV_R2['Alltime'],geo_month_CV_R2=geophysical_month_CV_R2['Alltime'],
                CV_slope=CV_slope['Alltime'],annual_CV_slope=annual_CV_slope['Alltime'],month_CV_slope=month_CV_slope['Alltime'],
                CV_RMSE=CV_RMSE['Alltime'], annual_CV_RMSE=annual_CV_RMSE['Alltime'], month_CV_RMSE=month_CV_RMSE['Alltime'],
                beginyear='Alltime', endyear='Alltime', species=species, kfold=kfold, repeats=repeats)
    for imodel in range(len(beginyears)):
        output_text(outfile=txt_outfile, status='a', CV_R2=CV_R2[str(beginyears[imodel])], annual_CV_R2=annual_CV_R2[str(beginyears[imodel])], month_CV_R2=month_CV_R2[str(beginyears[imodel])],
                    training_annual_CV_R2=training_annual_CV_R2[str(beginyears[imodel])], training_month_CV_R2=training_month_CV_R2[str(beginyears[imodel])],
                    geo_annual_CV_R2=geophysical_annual_CV_R2[str(beginyears[imodel])],geo_month_CV_R2=geophysical_month_CV_R2[str(beginyears[imodel])],
                    CV_slope=CV_slope[str(beginyears[imodel])], annual_CV_slope=annual_CV_slope[str(beginyears[imodel])], month_CV_slope=month_CV_slope[str(beginyears[imodel])],
                    CV_RMSE=CV_RMSE[str(beginyears[imodel])],annual_CV_RMSE=annual_CV_RMSE[str(beginyears[imodel])],month_CV_RMSE=month_CV_RMSE[str(beginyears[imodel])],
                    beginyear=beginyears[imodel], endyear=endyears[imodel], species=species, kfold=kfold, repeats=repeats)
    with open(txt_outfile,'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Training Variables: {}'.format(channel_names)])
    MONTH = ['01', '02', '03', '04', '05', '06','07', '08', '09', '10', '11', '12']
    
    # plot loss and accuracy vs epoch for for fold, one model
    save_loss_accuracy(model_outdir=model_outdir,loss=train_loss,accuracy=train_acc, valid_loss=valid_losses, valid_accuracy=test_acc,typeName=typeName
                       ,version=version,species=species,nchannel=nchannel, special_name=special_name, width=width, height=height)
    
    #plot_save_loss_accuracy_figure(loss=train_loss, accuracy=train_acc, typeName=typeName, species=species, version=version, nchannel=nchannel, width=width, height=height, special_name=special_name)

    # Save and plot monthly data
    for imonth in range(len(MONTH)):
        for imodel in range(len(beginyears)):
            save_data_recording(obs_data=monthly_obs_data_recording[str(beginyears[imodel])][MONTH[imonth]],final_data=monthly_final_data_recording[str(beginyears[imodel])][MONTH[imonth]],
                                species=species,version=version,typeName=typeName, beginyear=beginyears[imodel],MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height)
        save_data_recording(obs_data=monthly_obs_data_recording['Alltime'][MONTH[imonth]],final_data=monthly_final_data_recording['Alltime'][MONTH[imonth]],
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height)
        
    # Save and plot annual data
    for imodel in range(len(beginyears)):
        save_data_recording(obs_data=annual_obs_data_recording[str(beginyears[imodel])],final_data=annual_final_data_recording[str(beginyears[imodel])],
                                species=species,version=version,typeName=typeName, beginyear=beginyears[imodel],MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height)
    save_data_recording(obs_data=annual_obs_data_recording['Alltime'],final_data=annual_final_data_recording['Alltime'],
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height)
           

    return
