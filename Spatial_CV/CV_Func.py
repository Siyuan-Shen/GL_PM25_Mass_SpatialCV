import torch
import numpy as np
import torch.nn as nn
import os
import gc

from sklearn.model_selection import RepeatedKFold
from Spatial_CV.Model_Func import predict, train, weight_reset
from Spatial_CV.Statistic_Func import linear_regression, regress2, Cal_RMSE, Calculate_PWA_PM25
from Spatial_CV.Net_Construction import  ResNet, BasicBlock, Bottleneck, Net
from Spatial_CV.visualization import plot_loss_accuracy_with_epoch,regression_plot, bias_regression_plot,PM25_histgram_distribution_plot,regression_plot_area_test_average,PM25_histgram_distribution_area_tests_plot,regression_plot_ReducedAxisReduced
from Spatial_CV.ConvNet_Data import normalize_Func, Normlize_Training_Datasets, Normlize_Testing_Datasets, Data_Augmentation, Get_GeophysicalPM25_Datasets
from Spatial_CV.data_func import initialize_AVD_DataRecording, calculate_Statistics_results, get_longterm_array
from Spatial_CV.utils import *
from Spatial_CV.iostram import output_text, save_loss_accuracy
from .Model_Func import MyLoss,initialize_weights_kaiming,weight_init_normal
import random
import csv




def GetAreaCVResults(Area_Name:str,version:str,special_name:str,Extent,test_index, sites_index, Y_index,
                     test_obs_data:np.array, final_data:np.array,
                     beginyear:int, endyear:int,count:int,channel:int):
    '''
    :param Extent: The array for area. [bottom_lat, up_lat, left_lon, right_lon]
    :param test_index: The test index array for this fold.
    :param sites_index: The initial index for sites. Here is np.array(range(10870))
    :param Y_index: The index for tested sites for all months.
    :param test_obs_data: The observation datasets for test datasets.
    :param final_data: The final prediction datasets for test datasets.
    :param beginyear: The begin year.
    :param endyear: The end year.
    :return:
    '''

    # *----------------------------*
    # Get the area index and the area index for all months.
    # *----------------------------*
    area_test_index = get_area_index(extent=Extent, test_index=test_index)
    area_index = np.zeros((12 * (endyear - beginyear + 1) * len(area_test_index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        area_index[i * len(area_test_index):(i + 1) * len(area_test_index)] = ((beginyear - 1998) * 12 + i) * 10870 + \
                                                                        sites_index[area_test_index]
    
    ## Get the position of area index of the Y_index
    area_index_of_TestIndex = np.where(area_index == Y_index[:, None])[0]

    ## Get the observation data and predicted data for this area.
    area_obs_test = test_obs_data[area_index_of_TestIndex]
    area_final_test = final_data[area_index_of_TestIndex]

    # *----------------------------*
    # Initialize the arries for the annual results and month results. To calculate the R2 for each month and annually averaged data.
    # *----------------------------*
    month = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    area_annual_mean_obs = np.zeros(len(area_test_index) * (endyear - beginyear + 1), dtype=np.float64)
    area_annual_predict  = np.zeros(len(area_test_index) * (endyear - beginyear + 1), dtype=np.float64)
    area_month_obs = np.zeros((endyear - beginyear + 1) * len(area_test_index), dtype=np.float64)
    area_month_predict = np.zeros((endyear - beginyear + 1) * len(area_test_index), dtype=np.float64)
    month_CV_R2 = np.zeros(12)

    # *----------------------------*
    # Get the annually averaged observation data and predicted data for each sites located in the selected area.
    # *----------------------------*
    for iy in range((endyear - beginyear + 1)):
        for isite in range(len(area_test_index)):
            area_annual_mean_obs[isite + iy * len(area_test_index)] = np.mean(
                area_obs_test[isite + (iy * 12 + month) * len(area_test_index)])
            area_annual_predict[isite + iy * len(area_test_index)] = np.mean(
                area_final_test[isite + (iy * 12 + month) * len(area_test_index)])
    regression_plot_area_test_average(plot_obs_pm25=area_annual_mean_obs,plot_pre_pm25=area_annual_predict,
                                      version=version,channel=channel,special_name=special_name,
                                      area_name=Area_Name,extentlim=120,time='Annual',fold=count)
    PM25_histgram_distribution_area_tests_plot(plot_obs_pm25=area_annual_mean_obs,plot_pre_pm25=area_annual_predict,
                                      version=version,channel=channel,special_name=special_name,
                                      area_name=Area_Name,time='Annual',fold=count,range=(0,100),bins=100)
    # *----------------------------*
    # Seperate the observation and prediction for each month.
    # *----------------------------*
    month_name = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for imonth in range(len(month)):
        for iy in range((endyear - beginyear + 1)):
            area_month_obs[iy * len(area_test_index):(iy + 1) * len(area_test_index)] = \
                area_obs_test[(iy * 12 + imonth) * len(area_test_index):(iy * 12 + imonth + 1) * len(area_test_index)]
            area_month_predict[iy * len(area_test_index):(iy + 1) * len(area_test_index)] = \
                area_final_test[(iy * 12 + imonth) * len(area_test_index):(iy * 12 + imonth + 1) * len(area_test_index)]

        regression_plot_area_test_average(plot_obs_pm25=area_month_obs,plot_pre_pm25=area_month_predict,version=version,
                                          channel=channel,special_name=special_name,area_name=Area_Name,extentlim=120,
                                          time=month_name[imonth],fold=count)
        PM25_histgram_distribution_area_tests_plot(plot_obs_pm25=area_month_obs,plot_pre_pm25=area_month_predict,version=version,
                                          channel=channel,special_name=special_name,area_name=Area_Name,
                                          time=month_name[imonth],fold=count,range=(0,100),bins=100)
        month_CV_R2[imonth] = linear_regression(area_month_obs, area_month_predict)

    annual_R2 = linear_regression(area_annual_mean_obs, area_annual_predict)


    return annual_R2, month_CV_R2, area_obs_test, area_final_test



def MultiyearAreaModelCrossValid(train_input, true_input,channel_index, kfold:int, repeats:int,
                         extent:np.array,num_epochs:int, batch_size:int, learning_rate:float,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array, bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,
                         Log_PM25:bool):

    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    site_index = np.array(range(10870))         ### The index of sites.
    nchannel   = len(channel_index)    ### The number of channels.
    width      = train_input.shape[2]    ### The width of the input images.
    count      = 0                       ### Initialize the count number.
    seed = Get_CV_seed()                 ### Get the seed for random numbers for the folds seperation.
    ### Get observation data and Normalized parameters
    obs_data, obs_mean, obs_std = Get_data_NormPara(input_dir='/my-projects/Projects/MLCNN_PM25_2021/data/',input_file='obsPM25.npy')
    geo_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/geoPM25.npy')
    ### Initialize the CV R2 arrays for all datasets
    CV_R2, annual_CV_R2, month_CV_R2, CV_slope, annual_CV_slope, month_CV_slope,CV_RMSE, annual_CV_RMSE, month_CV_RMSE = Initialize_CV_Dic(kfold=kfold,repeats=repeats,breakpoints=beginyear)
    

    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    annual_final_dic, annual_obs_dic = Initialize_DataRecording_Dic(breakpoints=beginyear)

    train_input,train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index)
    train_input = train_input[:,channel_index,:,:]
    GeoPM25_mean = train_mean[16,int((width-1)/2),int((width-1)/2)]
    GeoPM25_std  = train_std[16,int((width-1)/2),int((width-1)/2)]

    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    
    for train_index, test_index in rkf.split(site_index):
        # *------------------------------------------------------------------------------*#
        ## Initialize the results arraies.
        ## For recording all models results in this fold.
        # *------------------------------------------------------------------------------*#
        overall_final_test = np.array([],dtype = np.float64)
        overall_obs_test   = np.array([],dtype = np.float64)
        for imodel in range(len(beginyear)):
            area_test_index = get_area_index(extent=extent, test_index=test_index)
            Y_index, X_index = GetXYIndex(Global_index=site_index,area_index=area_test_index,train_index=train_index,
                                          beginyear=beginyear[imodel],endyear=endyear[imodel],databeginyear=databeginyear,
                                          GLsitesNum=len(site_index))
            
            #X_train = Normlize_Training_Datasets(train_input=train_input[X_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #y_train = Normlize_Training_Datasets(train_input=train_input[Y_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #X_test,obs_mean,obs_std = Normlize_Testing_Datasets(true_input=true_input[X_index])  # Area Normalize True
            #X_test = true_input[X_index]

            X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
            y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]

            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            #cnn_model = ResNet(nchannel=nchannel,block=BasicBlock,blocks_num=[1,1,1,1],num_classes=1,include_top=True,
            #groups=1,width_per_group=width)
            cnn_model = Net(nchannel=nchannel)
            #cnn_model.apply(initialize_weights_Xavier) # No need for Residual Net

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)

            
            train_loss, train_acc = train(cnn_model, X_train, X_test,batch_size,learning_rate, num_epochs,GeoPM25_mean=GeoPM25_mean,GeoPM25_std=GeoPM25_std)
            X_train_aug = []
            X_test_aug = []
            # *------------------------------------------------------------------------------*#
            ## Save Model results.
            # *------------------------------------------------------------------------------*#
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            modelfile = model_outdir + 'CNN_PM25_Spatial_'+typeName+'_'+Area + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
            torch.save(cnn_model, modelfile)

            # *------------------------------------------------------------------------------*#
            ## Validation Process
            # *------------------------------------------------------------------------------*#
            Validation_Prediction = predict(y_train, cnn_model, width, 3000)
            if bias == True:
                final_data = Validation_Prediction + geo_data[Y_index]
            elif Normlized_PM25 == True:
                final_data = Validation_Prediction * obs_std + obs_mean
            elif Absolute_Pm25 == True:
                final_data = Validation_Prediction
            elif Log_PM25 == True:
                final_data = np.exp(Validation_Prediction) - 1

            # *------------------------------------------------------------------------------*#
            ## Recording Results
            # *------------------------------------------------------------------------------*#
            test_obs_data = obs_data[Y_index]
            overall_final_test = np.append(overall_final_test,final_data)
            overall_obs_test   = np.append(overall_obs_test,test_obs_data)


            # *------------------------------------------------------------------------------*#
            ## Calculate the correlation R2 for this model this fold
            # *------------------------------------------------------------------------------*#
            print('Area: ',Area,' fold:', str(count), ' beginyear: ', str(beginyear[imodel]),' endyear: ', str(endyear[imodel]))
            CV_R2[str(beginyear[imodel])][count] = linear_regression(final_data,test_obs_data)
            annual_R2,annual_final_data,annual_mean_obs = CalculateAnnualR2(test_index=area_test_index,final_data=final_data,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
            annual_final_dic[str(beginyear[imodel])] = np.append(annual_final_dic[str(beginyear[imodel])],annual_final_data)
            annual_obs_dic[str(beginyear[imodel])] = np.append(annual_obs_dic[str(beginyear[imodel])],annual_mean_obs)
            annual_CV_R2[str(beginyear[imodel])][count] = annual_R2
            month_R2 = CalculateMonthR2(test_index=area_test_index,final_data=final_data,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
            month_CV_R2[str(beginyear[imodel])][:,count] = month_R2

        # *------------------------------------------------------------------------------*#
        ## Calculate the correlation R2 for all models this fold
        # *------------------------------------------------------------------------------*#
        print('Area: ',Area, ' fold: ',str(count),  ' - Alltime')
        CV_R2['Alltime'][count] = linear_regression(overall_final_test, overall_obs_test)
        annual_R2, annual_final_data, annual_mean_obs = CalculateAnnualR2(test_index=area_test_index, final_data=overall_final_test,
                                                                          test_obs_data=overall_obs_test,
                                                                          beginyear=beginyear[0],
                                                                          endyear=endyear[-1])
        annual_final_dic['Alltime'] = np.append(annual_final_dic['Alltime'], annual_final_data)
        annual_obs_dic['Alltime'] = np.append(annual_obs_dic['Alltime'],annual_mean_obs)
        annual_CV_R2['Alltime'][count] = annual_R2
        month_R2 = CalculateMonthR2(test_index=area_test_index, final_data=overall_final_test,
                                    test_obs_data=overall_obs_test,
                                    beginyear=beginyear[0],
                                    endyear=endyear[-1])
        month_CV_R2['Alltime'][:, count] = month_R2

        count += 1
    
    txt_outdir = txt_dir + '{}/Results/results-SpatialCV/'.format(version)
    if not os.path.isdir(txt_outdir):
        os.makedirs(txt_outdir)
    txtoutfile = txt_outdir + 'Spatial_CV_'+ typeName +'_v' + version + '_' + str(nchannel) + 'Channel_'+Area+'_' + str(width) + 'x' + str(width) + special_name + '.csv'

    for imodel in range(len(beginyear)):
        if imodel == 0:
            status = 'w'
        else:
            status = 'a'
        Output_Text(outfile=txtoutfile,status=status,CV_R2=CV_R2[str(beginyear[imodel])],annual_CV_R2=annual_CV_R2[str(beginyear[imodel])],
                    month_CV_R2=month_CV_R2[str(beginyear[imodel])],beginyear=beginyear[imodel],endyear=endyear[imodel],Area=Area,
                    kfold=kfold,repeats=repeats)
        regression_plot(plot_obs_pm25=annual_obs_dic[str(beginyear[imodel])],plot_pre_pm25=annual_final_dic[str(beginyear[imodel])],
                        version=version,channel=nchannel,special_name=special_name,area_name=Area,beginyear=str(beginyear[imodel]),
                        endyear=str(endyear[imodel]),extentlim=4.2*np.mean(annual_obs_dic[str(beginyear[imodel])]),
                        bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)

    Output_Text(outfile=txtoutfile, status='a', CV_R2=CV_R2['Alltime'],
                annual_CV_R2=annual_CV_R2['Alltime'],
                month_CV_R2=month_CV_R2['Alltime'], beginyear='Alltime',
                endyear=' ',Area=Area,
                kfold=kfold, repeats=repeats)
    regression_plot(plot_obs_pm25=annual_obs_dic['Alltime'],
                    plot_pre_pm25=annual_final_dic['Alltime'],
                    version=version, channel=nchannel, special_name=special_name, area_name=Area,
                    beginyear='Alltime',
                    endyear='', extentlim=2.2 * np.mean(annual_obs_dic['Alltime']),
                     bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)
    del final_data, overall_final_test, obs_data, overall_obs_test,train_input, true_input
    gc.collect()

    return 

def MultiyearMultiAreasSpatialCrossValidation(train_input, true_input,channel_index, kfold:int, repeats:int,
                         extent,num_epochs:int, batch_size:int, learning_rate:float,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array,bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,
                         Log_PM25:bool):

    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    site_index = np.array(range(10870))         ### The index of sites.
    nchannel   = len(channel_index)    ### The number of channels.
    width      = train_input.shape[2]    ### The width of the input images.
    count      = 0                       ### Initialize the count number.
    seed = Get_CV_seed()                 ### Get the seed for random numbers for the folds seperation.
    ### Get observation data and Normalized parameters
    obs_data, obs_mean, obs_std = Get_data_NormPara(input_dir='/my-projects/Projects/MLCNN_PM25_2021/data/',input_file='obsPM25.npy')
    geo_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/geoPM25.npy')
    population_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/CoMonitors_Population_Data.npy')
    ### Initialize the CV R2 arrays for all datasets
    extent_dic = extent_table()
    MultiyearForMultiAreasList = MultiyearForMultiAreasLists ## Each model test on which areas
    Area_beginyears = {'NA':NA_beginyear,'EU':EU_beginyear,'AS':AS_beginyear,'GL':GL_beginyear}
    Areas = ['NA','EU','AS','GL']## Alltime areas names.
    CV_R2, annual_CV_R2, month_CV_R2,CV_slope, annual_CV_slope, month_CV_slope, CV_RMSE, annual_CV_RMSE, month_CV_RMSE,annual_CV_PWAModel,month_CV_PWAModel,annual_CV_PWAMonitor,month_CV_PWAMonitor = Initialize_multiareas_CV_Dic(kfold=kfold,repeats=repeats,breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    annual_final_dic, annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)

    train_input,train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index)
    
    GeoPM25_mean = train_mean[16,int((width-1)/2),int((width-1)/2)]
    GeoPM25_std  = train_std[16,int((width-1)/2),int((width-1)/2)]
    SitesNumber_mean = train_mean[31,int((width-1)/2),int((width-1)/2)]
    SitesNumber_std  = train_std[31,int((width-1)/2),int((width-1)/2)]
    train_input = train_input[:,channel_index,:,:]

    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    Allfolds_final_test = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
    Allfolds_obs_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
    Allfolds_population_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
    for train_index, test_index in rkf.split(site_index):
        # *------------------------------------------------------------------------------*#
        ## Initialize the results arraies.
        ## For recording all models results in this fold.
        # *------------------------------------------------------------------------------*#
        overall_final_test = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        overall_obs_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        overall_population_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        for imodel in range(len(beginyear)):

            X_index = GetTrainingIndex(Global_index=site_index,train_index=train_index,beginyear=beginyear[imodel],
                                            endyear=endyear[imodel],databeginyear=databeginyear,GLsitesNum=len(site_index))
            print('Train Index length: ', len(train_index),'\n X_index length: ', len(X_index))
            #X_train = Normlize_Training_Datasets(train_input=train_input[X_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #y_train = Normlize_Training_Datasets(train_input=train_input[Y_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #X_test,obs_mean,obs_std = Normlize_Testing_Datasets(true_input=true_input[X_index])  # Area Normalize True
            #X_test = true_input[X_index]

            X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
            
            

            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            #cnn_model = ResNet(nchannel=nchannel,block=BasicBlock,blocks_num=[1,1,1,1],num_classes=1,include_top=True,
            #groups=1,width_per_group=width)
            cnn_model = Net(nchannel=nchannel)
            #cnn_model.apply(initialize_weights_Xavier) # No need for Residual Net

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)

            train_loss, train_acc = train(cnn_model, X_train, X_test,batch_size,learning_rate, num_epochs,GeoPM25_mean=GeoPM25_mean,GeoPM25_std=GeoPM25_std,SitesNumber_mean=SitesNumber_mean,SitesNumber_std=SitesNumber_std)
                
            # *------------------------------------------------------------------------------*#
            ## Save Model results.
            # *------------------------------------------------------------------------------*#
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            modelfile = model_outdir + 'CNN_PM25_Spatial_'+typeName+'_'+Area+'_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
            torch.save(cnn_model, modelfile)

            #if ForcedUnitySlope == True:
            #    Training_Estimation = predict(X_train,cnn_model,width,3000)
            #    if bias == True:
            #        train_final_data = Training_Estimation + geo_data[X_index]
            #    elif Normlized_PM25 == True:
            #        train_final_data = Training_Estimation * obs_std + obs_mean
            #    elif Absolute_Pm25 == True:
            #        train_final_data = Training_Estimation
            #    elif Log_PM25 == True:
            #        train_final_data = np.exp(Training_Estimation) - 1
            #    regression_Dic = regress2(_x=X_test,_y=train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            #    offset,slope = regression_Dic['intercept'], regression_Dic['slope']
             
                

                
            for iarea in range(len(MultiyearForMultiAreasList[imodel])):
                extent = extent_dic[MultiyearForMultiAreasList[imodel][iarea]]
                area_test_index = get_area_index(extent=extent, test_index=test_index)
                Y_index = GetValidationIndex(area_index=area_test_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]

                 # *------------------------------------------------------------------------------*#
                ## Validation Process
                # *------------------------------------------------------------------------------*#
                Validation_Prediction = predict(y_train, cnn_model, width, 3000)
                
                if bias == True:
                    final_data = Validation_Prediction + geo_data[Y_index]
                elif Normlized_PM25 == True:
                    final_data = Validation_Prediction * obs_std + obs_mean
                elif Absolute_Pm25 == True:
                    final_data = Validation_Prediction
                elif Log_PM25 == True:
                    final_data = np.exp(Validation_Prediction) - 1
                # *------------------------------------------------------------------------------*#
                ## Recording Results
                # *------------------------------------------------------------------------------*#
                test_obs_data = obs_data[Y_index]
                Validation_population = population_data[Y_index]
                overall_final_test[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_final_test[MultiyearForMultiAreasList[imodel][iarea]],final_data)
                overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]],test_obs_data)
                overall_population_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_population_test[MultiyearForMultiAreasList[imodel][iarea]],Validation_population)
                Allfolds_final_test[MultiyearForMultiAreasList[imodel][iarea]] = np.append(Allfolds_final_test[MultiyearForMultiAreasList[imodel][iarea]],final_data)
                Allfolds_obs_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(Allfolds_obs_test[MultiyearForMultiAreasList[imodel][iarea]],test_obs_data)
                Allfolds_population_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(Allfolds_population_test[MultiyearForMultiAreasList[imodel][iarea]],Validation_population)
                
                # *------------------------------------------------------------------------------*#
                ## Calculate the correlation R2 for this model this fold
                # *------------------------------------------------------------------------------*#
                print('Area: ',Areas[iarea],' fold:', str(count), ' beginyear: ', str(beginyear[imodel]),' endyear: ', str(endyear[imodel]))
                CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = linear_regression(final_data,test_obs_data)
                CV_regression_Dic = regress2(_x=test_obs_data,_y=final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = CV_regression_Dic['slope']
                CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = Cal_RMSE(test_obs_data,final_data)

                annual_R2,annual_final_data,annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2(test_index=area_test_index,final_data=final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_final_dic[str(beginyear[imodel])],annual_final_data)
                annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_obs_dic[str(beginyear[imodel])],annual_mean_obs)
                annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_R2
                annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_slope
                annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_RMSE
                annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_model
                annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_monitor

                month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2(test_index=area_test_index,final_data = final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_R2
                month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_slope
                month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_RMSE
                month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_model
                month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_monitor
        # *------------------------------------------------------------------------------*#
        ## Calculate the correlation R2 for all models this fold
        # *------------------------------------------------------------------------------*#
        
        for iarea in range(len(Areas)):
            extent = extent_dic[Areas[iarea]]
            area_test_index = get_area_index(extent=extent, test_index=test_index)
            print('Area: ',Areas[iarea], ' fold: ',str(count),  ' - Alltime')
            CV_R2['Alltime'][Areas[iarea]][count] = linear_regression(overall_final_test[Areas[iarea]], overall_obs_test[Areas[iarea]])
            CV_regression_Dic = regress2(_x=overall_obs_test[Areas[iarea]],_y=overall_final_test[Areas[iarea]],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            CV_slope['Alltime'][Areas[iarea]][count] = CV_regression_Dic['slope']
            CV_RMSE['Alltime'][Areas[iarea]][count] = Cal_RMSE(overall_obs_test[Areas[iarea]],overall_final_test[Areas[iarea]])

            annual_R2, annual_final_data, annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2(test_index=area_test_index, final_data=overall_final_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                                                          test_obs_data=overall_obs_test[Areas[iarea]],
                                                                          beginyear=Area_beginyears[Areas[iarea]],
                                                                          endyear=endyear[-1])
            annual_final_dic['Alltime'][Areas[iarea]] = np.append(annual_final_dic['Alltime'][Areas[iarea]], annual_final_data)
            annual_obs_dic['Alltime'][Areas[iarea]] = np.append(annual_obs_dic['Alltime'][Areas[iarea]],annual_mean_obs)
            annual_CV_R2['Alltime'][Areas[iarea]][count] = annual_R2
            annual_CV_slope['Alltime'][Areas[iarea]][count] = annual_slope
            annual_CV_RMSE['Alltime'][Areas[iarea]][count] = annual_RMSE
            annual_CV_PWAModel['Alltime'][Areas[iarea]][count] = annual_PWA_model
            annual_CV_PWAMonitor['Alltime'][Areas[iarea]][count] = annual_PWA_monitor

            month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2(test_index=area_test_index, final_data=overall_final_test[Areas[iarea]],
                                    test_obs_data=overall_obs_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                    beginyear=Area_beginyears[Areas[iarea]],
                                    endyear=endyear[-1])
            month_CV_R2['Alltime'][Areas[iarea]][:, count] = month_R2
            month_CV_slope['Alltime'][Areas[iarea]][:, count] = month_slope
            month_CV_RMSE['Alltime'][Areas[iarea]][:, count] = month_RMSE
            month_CV_PWAModel['Alltime'][Areas[iarea]][:, count] = month_PWA_model
            month_CV_PWAMonitor['Alltime'][Areas[iarea]][:, count] = month_PWA_monitor

        count += 1
    txt_outdir = txt_dir + '{}/Results/results-SpatialCV/'.format(version)
    if not os.path.isdir(txt_outdir):
        os.makedirs(txt_outdir)
    txtoutfile = txt_outdir + 'Spatial_CV_'+ typeName +'_v' + version + '_' + str(nchannel) + 'Channel_' + str(width) + 'x' + str(width) + special_name + '.csv'

    for imodel in range(len(beginyear)):
        if imodel == 0:
            status = 'w'
        else:
            status = 'a'
        for iarea in range(len(MultiyearForMultiAreasList[imodel])):
            Output_Text(outfile=txtoutfile,status=status,CV_R2=CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_R2=annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_R2=month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_slope=CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_slope=annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_slope=month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_RMSE=CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_RMSE=annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_RMSE=month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_models=annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_monitors=annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_models=month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_monitors=month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    beginyear=beginyear[imodel],endyear=endyear[imodel],Area=MultiyearForMultiAreasList[imodel][iarea],
                    kfold=kfold,repeats=repeats)
            #regression_plot(plot_obs_pm25=annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],plot_pre_pm25=annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
            #            version=version,channel=nchannel,special_name=special_name,area_name=Area,beginyear=str(beginyear[imodel]),
            #            endyear=str(endyear[imodel]),extentlim=4.2*np.mean(annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]]),
            #            bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
            #             Log_PM25=Log_PM25)
    for iarea in range(len(Areas)):
        Output_Text(outfile=txtoutfile, status='a', CV_R2=CV_R2['Alltime'][Areas[iarea]],
                annual_CV_R2=annual_CV_R2['Alltime'][Areas[iarea]],
                month_CV_R2=month_CV_R2['Alltime'][Areas[iarea]],
                CV_slope=CV_slope['Alltime'][Areas[iarea]],
                annual_CV_slope=annual_CV_slope['Alltime'][Areas[iarea]],
                month_CV_slope=month_CV_slope['Alltime'][Areas[iarea]],
                CV_RMSE=CV_RMSE['Alltime'][Areas[iarea]],
                annual_CV_RMSE=annual_CV_RMSE['Alltime'][Areas[iarea]],
                month_CV_RMSE=month_CV_RMSE['Alltime'][Areas[iarea]], 
                annual_CV_models=annual_CV_PWAModel['Alltime'][Areas[iarea]], 
                annual_CV_monitors=annual_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                month_CV_models=month_CV_PWAModel['Alltime'][Areas[iarea]], 
                month_CV_monitors=month_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                beginyear='Alltime',
                endyear=' ',Area=Areas[iarea],
                kfold=kfold, repeats=repeats)
        regression_plot(plot_obs_pm25=annual_obs_dic['Alltime'][Areas[iarea]],
                    plot_pre_pm25=annual_final_dic['Alltime'][Areas[iarea]],
                    version=version, channel=nchannel, special_name=special_name, area_name=Areas[iarea],
                    beginyear='Alltime',
                    endyear='', extentlim=2.2 * np.mean(annual_obs_dic['Alltime'][Areas[iarea]]),
                     bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)
    del final_data, overall_final_test, obs_data, overall_obs_test,train_input, true_input
    gc.collect()

    return txtoutfile


def EachAreaForcedSlope_MultiyearMultiAreasSpatialCrossValidation(train_input, true_input,channel_index, kfold:int, repeats:int,
                         extent,num_epochs:int, batch_size:int, learning_rate:float,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array, bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,EachMonthSlopeUnity:bool,
                         EachAreaForcedSlopeUnity:bool,
                         Log_PM25:bool, ):

    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    site_index = np.array(range(10870))         ### The index of sites.
    nchannel   = len(channel_index)    ### The number of channels.
    width      = train_input.shape[2]    ### The width of the input images.
    count      = 0                       ### Initialize the count number.
    seed = Get_CV_seed()                 ### Get the seed for random numbers for the folds seperation.
    ### Get observation data and Normalized parameters
    obs_data, obs_mean, obs_std = Get_data_NormPara(input_dir='/my-projects/Projects/MLCNN_PM25_2021/data/',input_file='obsPM25.npy')
    geo_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/geoPM25.npy')
    population_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/CoMonitors_Population_Data.npy')
    ### Initialize the CV R2 arrays for all datasets
    extent_dic = extent_table()
    
    MultiyearForMultiAreasList = MultiyearForMultiAreasLists ## Each model test on which areas
    Area_beginyears = {'NA':NA_beginyear,'EU':EU_beginyear,'AS':AS_beginyear,'GL':GL_beginyear}
    #MultiyearForMultiAreasList = [['NA','EU','AS','GL']]## Each model test on which areas
    #Area_beginyears = {'NA':2015,'EU':2015,'AS':2015,'GL':2015}
    
    Areas = ['NA','EU','AS','GL']## Alltime areas names.
    CV_R2, annual_CV_R2, month_CV_R2,CV_slope, annual_CV_slope, month_CV_slope, CV_RMSE, annual_CV_RMSE, month_CV_RMSE,annual_CV_PWAModel,month_CV_PWAModel,annual_CV_PWAMonitor,month_CV_PWAMonitor = Initialize_multiareas_CV_Dic(kfold=kfold,repeats=repeats,breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
     # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    annual_final_dic, annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)

    train_input,train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index)
    
    GeoPM25_mean = train_mean[16,int((width-1)/2),int((width-1)/2)]
    GeoPM25_std  = train_std[16,int((width-1)/2),int((width-1)/2)]
    SitesNumber_mean = train_mean[31,int((width-1)/2),int((width-1)/2)]
    SitesNumber_std  = train_std[31,int((width-1)/2),int((width-1)/2)]
    train_input = train_input[:,channel_index,:,:]

    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    for train_index, test_index in rkf.split(site_index):
        # *------------------------------------------------------------------------------*#
        ## Initialize the results arraies.
        ## For recording all models results in this fold.
        # *------------------------------------------------------------------------------*#
        overall_final_test = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        overall_obs_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        overall_population_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        for imodel in range(len(beginyear)):

            X_index = GetTrainingIndex(Global_index=site_index,train_index=train_index,beginyear=beginyear[imodel],
                                            endyear=endyear[imodel],databeginyear=databeginyear,GLsitesNum=len(site_index))
            print('Train Index length: ', len(train_index),'\n X_index length: ', len(X_index))
            #X_train = Normlize_Training_Datasets(train_input=train_input[X_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #y_train = Normlize_Training_Datasets(train_input=train_input[Y_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #X_test,obs_mean,obs_std = Normlize_Testing_Datasets(true_input=true_input[X_index])  # Area Normalize True
            #X_test = true_input[X_index]

            X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
            
            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            cnn_model = ResNet(nchannel=nchannel,block=BasicBlock,blocks_num=[1,1,1,1],num_classes=1,include_top=True,
            groups=1,width_per_group=width)
            #cnn_model = Net(nchannel=nchannel)
            #cnn_model.apply(initialize_weights_Xavier) # No need for Residual Net

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)

            train_loss, train_acc = train(cnn_model, X_train, X_test,batch_size,learning_rate, num_epochs,GeoPM25_mean=GeoPM25_mean,GeoPM25_std=GeoPM25_std,SitesNumber_mean=SitesNumber_mean,SitesNumber_std=SitesNumber_std) 
            X_train_aug = []
            X_test_aug = []
            # *------------------------------------------------------------------------------*#
            ## Save Model results.
            # *------------------------------------------------------------------------------*#
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            modelfile = model_outdir + 'CNN_PM25_Spatial_'+typeName+'_'+Area+'_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
            torch.save(cnn_model, modelfile)

            #if ForcedUnitySlope == True:
            #    Training_Estimation = predict(X_train,cnn_model,width,3000)
            #    if bias == True:
            #        train_final_data = Training_Estimation + geo_data[X_index]
            #    elif Normlized_PM25 == True:
            #        train_final_data = Training_Estimation * obs_std + obs_mean
            #    elif Absolute_Pm25 == True:
            #        train_final_data = Training_Estimation
            #    elif Log_PM25 == True:
            #        train_final_data = np.exp(Training_Estimation) - 1
            #    regression_Dic = regress2(_x=X_test,_y=train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            #    offset,slope = regression_Dic['intercept'], regression_Dic['slope']                
            for iarea in range(len(MultiyearForMultiAreasList[imodel])):
                extent = extent_dic[MultiyearForMultiAreasList[imodel][iarea]]
                area_test_index = get_area_index(extent=extent, test_index=test_index)
                Y_index = GetValidationIndex(area_index=area_test_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]

                if EachAreaForcedSlopeUnity:
                    area_train_index = get_area_index(extent=extent, test_index=train_index)
                else:
                    area_train_index = get_area_index(extent=extent_dic['GL'], test_index=train_index)
                
                XforForcedSlope_index = GetValidationIndex(area_index=area_train_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                x_train_forSlope = train_input[XforForcedSlope_index,:,:,:]
                # *------------------------------------------------------------------------------*#
                ## Validation Process
                # *------------------------------------------------------------------------------*#
                Training_Prediction = predict(x_train_forSlope,cnn_model,width,3000)
                Validation_Prediction = predict(y_train, cnn_model, width, 3000)
                if bias == True:
                    final_data = Validation_Prediction + geo_data[Y_index]
                    train_final_data = Training_Prediction + geo_data[XforForcedSlope_index]
                elif Normlized_PM25 == True:
                    final_data = Validation_Prediction * obs_std + obs_mean
                    train_final_data = Training_Prediction * obs_std + obs_mean
                elif Absolute_Pm25 == True:
                    final_data = Validation_Prediction
                    train_final_data = Training_Prediction
                elif Log_PM25 == True:
                    final_data = np.exp(Validation_Prediction) - 1
                    train_final_data = np.exp(Training_Prediction) - 1
                
                final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=obs_data[XforForcedSlope_index],
                                              test_final_data=final_data,train_area_index=area_train_index,test_area_index=area_train_index,endyear=endyear[imodel],
                                              beginyear=beginyear[imodel],EachMonth=EachMonthSlopeUnity)
                
                
                
                # *------------------------------------------------------------------------------*#
                ## Recording Results
                # *------------------------------------------------------------------------------*#
                test_obs_data = obs_data[Y_index]
                Validation_population = population_data[Y_index]
                overall_final_test[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_final_test[MultiyearForMultiAreasList[imodel][iarea]],final_data)
                overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]],test_obs_data)
                overall_population_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_population_test[MultiyearForMultiAreasList[imodel][iarea]],Validation_population)
                # *------------------------------------------------------------------------------*#
                ## Calculate the correlation R2 for this model this fold
                # *------------------------------------------------------------------------------*#
                print('Area: ',Areas[iarea],' fold:', str(count), ' beginyear: ', str(beginyear[imodel]),' endyear: ', str(endyear[imodel]))
                CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = linear_regression(final_data,test_obs_data)
                CV_regression_Dic = regress2(_x=test_obs_data,_y=final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = CV_regression_Dic['slope']
                CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = Cal_RMSE(test_obs_data,final_data)

                annual_R2,annual_final_data,annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2(test_index=area_test_index,final_data=final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_final_dic[str(beginyear[imodel])],annual_final_data)
                annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_obs_dic[str(beginyear[imodel])],annual_mean_obs)
                annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_R2
                annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_slope
                annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_RMSE
                annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_model
                annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_monitor

                month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2(test_index=area_test_index,final_data = final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_R2
                month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_slope
                month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_RMSE
                month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_model
                month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_monitor
        # *------------------------------------------------------------------------------*#
        ## Calculate the correlation R2 for all models this fold
        # *------------------------------------------------------------------------------*#
        
        for iarea in range(len(Areas)):
            extent = extent_dic[Areas[iarea]]
            area_test_index = get_area_index(extent=extent, test_index=test_index)
            
            print('Area: ',Areas[iarea], ' fold: ',str(count),  ' - Alltime')
            CV_R2['Alltime'][Areas[iarea]][count] = linear_regression(overall_final_test[Areas[iarea]], overall_obs_test[Areas[iarea]])
            CV_regression_Dic = regress2(_x=overall_obs_test[Areas[iarea]],_y=overall_final_test[Areas[iarea]],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            CV_slope['Alltime'][Areas[iarea]][count] = CV_regression_Dic['slope']
            CV_RMSE['Alltime'][Areas[iarea]][count] = Cal_RMSE(overall_obs_test[Areas[iarea]],overall_final_test[Areas[iarea]])

            annual_R2, annual_final_data, annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2(test_index=area_test_index, final_data=overall_final_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                                                          test_obs_data=overall_obs_test[Areas[iarea]],
                                                                          beginyear=Area_beginyears[Areas[iarea]],
                                                                          endyear=endyear[-1])
            annual_final_dic['Alltime'][Areas[iarea]] = np.append(annual_final_dic['Alltime'][Areas[iarea]], annual_final_data)
            annual_obs_dic['Alltime'][Areas[iarea]] = np.append(annual_obs_dic['Alltime'][Areas[iarea]],annual_mean_obs)
            annual_CV_R2['Alltime'][Areas[iarea]][count] = annual_R2
            annual_CV_slope['Alltime'][Areas[iarea]][count] = annual_slope
            annual_CV_RMSE['Alltime'][Areas[iarea]][count] = annual_RMSE
            annual_CV_PWAModel['Alltime'][Areas[iarea]][count] = annual_PWA_model
            annual_CV_PWAMonitor['Alltime'][Areas[iarea]][count] = annual_PWA_monitor

            month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2(test_index=area_test_index, final_data=overall_final_test[Areas[iarea]],
                                    test_obs_data=overall_obs_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                    beginyear=Area_beginyears[Areas[iarea]],
                                    endyear=endyear[-1])
            month_CV_R2['Alltime'][Areas[iarea]][:, count] = month_R2
            month_CV_slope['Alltime'][Areas[iarea]][:, count] = month_slope
            month_CV_RMSE['Alltime'][Areas[iarea]][:, count] = month_RMSE
            month_CV_PWAModel['Alltime'][Areas[iarea]][:, count] = month_PWA_model
            month_CV_PWAMonitor['Alltime'][Areas[iarea]][:, count] = month_PWA_monitor

        count += 1
    txt_outdir = txt_dir + '{}/Results/results-SpatialCV/'.format(version)
    if not os.path.isdir(txt_outdir):
        os.makedirs(txt_outdir)
    txtoutfile = txt_outdir + 'Spatial_CV_'+ typeName +'_v' + version + '_' + str(nchannel) + 'Channel_' + str(width) + 'x' + str(width) + special_name + '.csv'

    for imodel in range(len(beginyear)):
        if imodel == 0:
            status = 'w'
        else:
            status = 'a'
        for iarea in range(len(MultiyearForMultiAreasList[imodel])):
            Output_Text(outfile=txtoutfile,status=status,CV_R2=CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_R2=annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_R2=month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_slope=CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_slope=annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_slope=month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_RMSE=CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_RMSE=annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_RMSE=month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_models=annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_monitors=annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_models=month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_monitors=month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    beginyear=beginyear[imodel],endyear=endyear[imodel],Area=MultiyearForMultiAreasList[imodel][iarea],
                    kfold=kfold,repeats=repeats)
            #regression_plot(plot_obs_pm25=annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],plot_pre_pm25=annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
            #            version=version,channel=nchannel,special_name=special_name,area_name=Area,beginyear=str(beginyear[imodel]),
            #            endyear=str(endyear[imodel]),extentlim=4.2*np.mean(annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]]),
            #            bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
            #             Log_PM25=Log_PM25)
    for iarea in range(len(Areas)):
        Output_Text(outfile=txtoutfile, status='a', CV_R2=CV_R2['Alltime'][Areas[iarea]],
                annual_CV_R2=annual_CV_R2['Alltime'][Areas[iarea]],
                month_CV_R2=month_CV_R2['Alltime'][Areas[iarea]],
                CV_slope=CV_slope['Alltime'][Areas[iarea]],
                annual_CV_slope=annual_CV_slope['Alltime'][Areas[iarea]],
                month_CV_slope=month_CV_slope['Alltime'][Areas[iarea]],
                CV_RMSE=CV_RMSE['Alltime'][Areas[iarea]],
                annual_CV_RMSE=annual_CV_RMSE['Alltime'][Areas[iarea]],
                month_CV_RMSE=month_CV_RMSE['Alltime'][Areas[iarea]], 
                annual_CV_models=annual_CV_PWAModel['Alltime'][Areas[iarea]], 
                annual_CV_monitors=annual_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                month_CV_models=month_CV_PWAModel['Alltime'][Areas[iarea]], 
                month_CV_monitors=month_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                beginyear='Alltime',
                endyear=' ',Area=Areas[iarea],
                kfold=kfold, repeats=repeats)
        regression_plot(plot_obs_pm25=annual_obs_dic['Alltime'][Areas[iarea]],
                    plot_pre_pm25=annual_final_dic['Alltime'][Areas[iarea]],
                    version=version, channel=nchannel, special_name=special_name, area_name=Areas[iarea],
                    beginyear='Alltime',
                    endyear='', extentlim=2.2 * np.mean(annual_obs_dic['Alltime'][Areas[iarea]]),
                     bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)
    del final_data, overall_final_test, obs_data, overall_obs_test,train_input, true_input
    gc.collect()

    return txtoutfile


def MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25(train_input, true_input,channel_index, kfold:int, repeats:int,
                         extent,num_epochs:int, batch_size:int, learning_rate:float,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array,bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,EachMonthSlopeUnity:bool,
                         EachAreaForcedSlopeUnity:bool,
                         Log_PM25:bool):

    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    site_index = np.array(range(10870))         ### The index of sites.
    nchannel   = len(channel_index)    ### The number of channels.
    width      = train_input.shape[2]    ### The width of the input images.
    count      = 0                       ### Initialize the count number.
    seed = Get_CV_seed()                 ### Get the seed for random numbers for the folds seperation.
    ### Get observation data and Normalized parameters
    obs_data, obs_mean, obs_std = Get_data_NormPara(input_dir='/my-projects/Projects/MLCNN_PM25_2021/data/',input_file='obsPM25.npy')
    geo_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/geoPM25.npy')
    population_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/CoMonitors_Population_Data.npy')
    ### Initialize the CV R2 arrays for all datasets
    extent_dic = extent_table()
    #MultiyearForMultiAreasList = [['NA'],['NA'],['NA','EU'],['NA','EU','AS','GL']]## Each model test on which areas
    #Area_beginyears = {'NA':2001,'EU':2010,'AS':2015,'GL':2015}
    MultiyearForMultiAreasList = MultiyearForMultiAreasLists ## Each model test on which areas
    Area_beginyears = {'NA':NA_beginyear,'EU':EU_beginyear,'AS':AS_beginyear,'GL':GL_beginyear}
    Areas = ['NA','EU','AS','GL']## Alltime areas names.
    training_CV_R2, training_annual_CV_R2,training_month_CV_R2, CV_R2, annual_CV_R2, month_CV_R2, CV_slope, annual_CV_slope, month_CV_slope, CV_RMSE, annual_CV_RMSE, month_CV_RMSE,annual_CV_PWAModel,month_CV_PWAModel,annual_CV_PWAMonitor,month_CV_PWAMonitor = Initialize_multiareas_optimalModel_CV_Dic(kfold=kfold,repeats=repeats,breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    annual_final_dic, annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
    training_annual_final_dic, training_annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)

    train_input,train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index)
    
    GeoPM25_mean = train_mean[16,int((width-1)/2),int((width-1)/2)]
    GeoPM25_std  = train_std[16,int((width-1)/2),int((width-1)/2)]
    SitesNumber_mean = train_mean[31,int((width-1)/2),int((width-1)/2)]
    SitesNumber_std  = train_std[31,int((width-1)/2),int((width-1)/2)]
    train_input = train_input[:,channel_index,:,:]

    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    for train_index, test_index in rkf.split(site_index):
        # *------------------------------------------------------------------------------*#
        ## Initialize the results arraies.
        ## For recording all models results in this fold.
        # *------------------------------------------------------------------------------*#
        overall_final_test = {}
        overall_geo_test   = {}
        overall_obs_test   = {}
        overall_train_final = {}
        overall_train_obs = {}
        overall_population_test = {}

        for iarea in Areas:
            overall_final_test[iarea] = np.array([],dtype = np.float64)
            overall_geo_test[iarea] = np.array([],dtype = np.float64)
            overall_obs_test[iarea] = np.array([],dtype = np.float64)
            overall_train_final[iarea] = np.array([],dtype = np.float64)
            overall_train_obs[iarea] = np.array([],dtype = np.float64)
            overall_population_test[iarea] = np.array([],dtype = np.float64)
        '''
        overall_final_test = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        overall_obs_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        overall_population_test   = {'NA':np.array([],dtype = np.float64),
                              'AS':np.array([],dtype = np.float64),
                              'EU':np.array([],dtype = np.float64),
                              'GL':np.array([],dtype = np.float64)}
        '''
        for imodel in range(len(beginyear)):

            X_index = GetTrainingIndex(Global_index=site_index,train_index=train_index,beginyear=beginyear[imodel],
                                            endyear=endyear[imodel],databeginyear=databeginyear,GLsitesNum=len(site_index))
            print('Train Index length: ', len(train_index),'\n X_index length: ', len(X_index))
            #X_train = Normlize_Training_Datasets(train_input=train_input[X_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #y_train = Normlize_Training_Datasets(train_input=train_input[Y_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #X_test,obs_mean,obs_std = Normlize_Testing_Datasets(true_input=true_input[X_index])  # Area Normalize True
            #X_test = true_input[X_index]

            X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
            
            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            cnn_model = ResNet(nchannel=nchannel,block=BasicBlock,blocks_num=[1,1,1,1],num_classes=1,include_top=True,
            groups=1,width_per_group=width)
            #cnn_model = Net(nchannel=nchannel)
            #cnn_model.apply(initialize_weights_Xavier) # No need for Residual Net

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)

            train_loss, train_acc = train(cnn_model, X_train, X_test,batch_size,learning_rate, num_epochs,GeoPM25_mean=GeoPM25_mean,GeoPM25_std=GeoPM25_std,SitesNumber_mean=SitesNumber_mean,SitesNumber_std=SitesNumber_std) 
            X_train_aug = []
            X_test_aug = []
            # *------------------------------------------------------------------------------*#
            ## Save Model results.
            # *------------------------------------------------------------------------------*#
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            modelfile = model_outdir + 'CNN_PM25_Spatial_'+typeName+'_'+Area+'_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
            torch.save(cnn_model, modelfile)

            #if ForcedUnitySlope == True:
            #    Training_Estimation = predict(X_train,cnn_model,width,3000)
            #    if bias == True:
            #        train_final_data = Training_Estimation + geo_data[X_index]
            #    elif Normlized_PM25 == True:
            #        train_final_data = Training_Estimation * obs_std + obs_mean
            #    elif Absolute_Pm25 == True:
            #        train_final_data = Training_Estimation
            #    elif Log_PM25 == True:
            #        train_final_data = np.exp(Training_Estimation) - 1
            #    regression_Dic = regress2(_x=X_test,_y=train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            #    offset,slope = regression_Dic['intercept'], regression_Dic['slope']                
            for iarea in range(len(MultiyearForMultiAreasList[imodel])):
                extent = extent_dic[MultiyearForMultiAreasList[imodel][iarea]]
                area_test_index = get_area_index(extent=extent, test_index=test_index)
                #GBD_area_index = load_GBD_area_index(area=MultiyearForMultiAreasList[imodel][iarea])
                #area_test_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=test_index)
                Y_index = GetValidationIndex(area_index=area_test_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]

                
                if EachAreaForcedSlopeUnity:
                    area_train_index = get_area_index(extent=extent, test_index=train_index)
                    area_train_forStatistic_index = get_area_index(extent=extent, test_index=train_index)
                    #area_train_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                    #area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                else:
                    area_train_index = get_area_index(extent=extent_dic['GL'], test_index=train_index)
                    area_train_forStatistic_index = get_area_index(extent=extent, test_index=train_index)
                    #GL_GBD_area_index = load_GBD_area_index(area='GL')
                    #area_train_index = get_test_index_inGBD_area(GBD_area_index=GL_GBD_area_index,test_index=train_index)
                    #area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                
                XforForcedSlope_index = GetValidationIndex(area_index=area_train_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                XforStatistic_index = GetValidationIndex(area_index=area_train_forStatistic_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                x_train_forSlope = train_input[XforForcedSlope_index,:,:,:]
                X_train_forStatistic = train_input[XforStatistic_index,:,:,:]


                # *------------------------------------------------------------------------------*#
                ## Validation Process
                # *------------------------------------------------------------------------------*#
                Training_Prediction = predict(x_train_forSlope,cnn_model,width,3000)
                Training_forStatistic = predict(X_train_forStatistic, cnn_model, width,3000)
                Validation_Prediction = predict(y_train, cnn_model, width, 3000)
                if bias == True:
                    final_data = Validation_Prediction + geo_data[Y_index]
                    train_final_data = Training_Prediction + geo_data[XforForcedSlope_index]
                    train_final_forStatistic = Training_forStatistic + geo_data[XforStatistic_index]
                elif Normlized_PM25 == True:
                    final_data = Validation_Prediction * obs_std + obs_mean
                    train_final_data = Training_Prediction * obs_std + obs_mean
                    train_final_forStatistic = Training_forStatistic* obs_std + obs_mean
                elif Absolute_Pm25 == True:
                    final_data = Validation_Prediction
                    train_final_data = Training_Prediction
                    train_final_forStatistic = Training_forStatistic
                elif Log_PM25 == True:
                    final_data = np.exp(Validation_Prediction) - 1
                    train_final_data = np.exp(Training_Prediction) - 1
                    train_final_forStatistic = np.exp(train_final_forStatistic) - 1
                nearest_distance = get_nearest_test_distance(area_test_index=area_test_index,area_train_index=train_index)
                coeficient = get_coefficients(nearest_site_distance=nearest_distance,beginyear=beginyear[imodel],
                                              endyear = endyear[imodel])
                final_data = (1.0-coeficient)*final_data + coeficient * geo_data[Y_index]
                print('Forced Slope Unity - length of area_test_index: ',len(area_test_index),' length of area_train_index',len(area_train_index),'Area: ',MultiyearForMultiAreasList[imodel][iarea],
                      '\nlength of final_data',len(final_data))
                final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=obs_data[XforForcedSlope_index],
                                              test_final_data=final_data,train_area_index=area_train_index,test_area_index=area_test_index,endyear=endyear[imodel],
                                              beginyear=beginyear[imodel],EachMonth=EachMonthSlopeUnity)
                
                
                
                # *------------------------------------------------------------------------------*#
                ## Recording Results
                # *------------------------------------------------------------------------------*#
                test_obs_data = obs_data[Y_index]
                Train_obs_data = obs_data[XforStatistic_index]
                Validation_population = population_data[Y_index]
                overall_final_test[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_final_test[MultiyearForMultiAreasList[imodel][iarea]],final_data)
                overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]],test_obs_data)
                overall_population_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_population_test[MultiyearForMultiAreasList[imodel][iarea]],Validation_population)
                overall_train_final[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_train_final[MultiyearForMultiAreasList[imodel][iarea]],train_final_forStatistic)
                overall_train_obs[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_train_obs[MultiyearForMultiAreasList[imodel][iarea]], Train_obs_data)
                
                # *------------------------------------------------------------------------------*#
                ## Calculate the correlation R2 for this model this fold
                # *------------------------------------------------------------------------------*#
                print('Area: ',Areas[iarea],' fold:', str(count), ' beginyear: ', str(beginyear[imodel]),' endyear: ', str(endyear[imodel]))
                CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = linear_regression(final_data,test_obs_data)
                CV_regression_Dic = regress2(_x=test_obs_data,_y=final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = CV_regression_Dic['slope']
                CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = Cal_RMSE(test_obs_data,final_data)

                annual_R2,annual_final_data,annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2_EachYear(test_index=area_test_index,final_data=final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel]) ###Each Year 
                training_annual_R2 = CalculateTrainingAnnualR2(train_index=area_train_forStatistic_index,train_final_data=train_final_forStatistic,train_obs_data=Train_obs_data,
                                                               beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_final_dic[str(beginyear[imodel])],annual_final_data)
                annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_obs_dic[str(beginyear[imodel])],annual_mean_obs)
                annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_R2
                annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_slope
                annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_RMSE
                annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_model
                annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_monitor
                training_annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = training_annual_R2
                
                training_month_R2 = CalculateTrainingMonthR2(train_index=area_train_forStatistic_index,final_training_data=train_final_forStatistic,
                                                             train_obs_data=Train_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2_EachYear(test_index=area_test_index,final_data = final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])###Each Year 
                month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_R2
                month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_slope
                month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_RMSE
                month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_model
                month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_monitor
                training_month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = training_month_R2

        # *------------------------------------------------------------------------------*#
        ## Calculate the correlation R2 for all models this fold
        # *------------------------------------------------------------------------------*#
        
        for iarea in range(len(Areas)):
            extent = extent_dic[Areas[iarea]]
            #GBD_area_index = load_GBD_area_index(area=Areas[iarea])
            #area_test_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index, test_index=test_index)
            #area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index, test_index=train_index)
            area_test_index = get_area_index(extent=extent,test_index=test_index)
            area_train_forStatistic_index = get_area_index(extent=extent, test_index=train_index)

            print('Area: ',Areas[iarea], ' fold: ',str(count),  ' - Alltime')
            CV_R2['Alltime'][Areas[iarea]][count] = linear_regression(overall_final_test[Areas[iarea]], overall_obs_test[Areas[iarea]])
            CV_regression_Dic = regress2(_x=overall_obs_test[Areas[iarea]],_y=overall_final_test[Areas[iarea]],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            CV_slope['Alltime'][Areas[iarea]][count] = CV_regression_Dic['slope']
            CV_RMSE['Alltime'][Areas[iarea]][count] = Cal_RMSE(overall_obs_test[Areas[iarea]],overall_final_test[Areas[iarea]])

            
            annual_R2, annual_final_data, annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2_EachYear(test_index=area_test_index, 
                                                                        final_data=overall_final_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                                                          test_obs_data=overall_obs_test[Areas[iarea]],
                                                                          beginyear=Area_beginyears[Areas[iarea]],
                                                                          endyear=endyear[-1])###Each Year 
            
            training_annual_R2 = CalculateTrainingAnnualR2(train_index=area_train_forStatistic_index,train_final_data=overall_train_final[Areas[iarea]],
                                                          train_obs_data=overall_train_obs[Areas[iarea]],
                                                            beginyear=Area_beginyears[Areas[iarea]],
                                                           endyear=endyear[-1])
            annual_final_dic['Alltime'][Areas[iarea]] = np.append(annual_final_dic['Alltime'][Areas[iarea]], annual_final_data)
            annual_obs_dic['Alltime'][Areas[iarea]] = np.append(annual_obs_dic['Alltime'][Areas[iarea]],annual_mean_obs)
            annual_CV_R2['Alltime'][Areas[iarea]][count] = annual_R2
            annual_CV_slope['Alltime'][Areas[iarea]][count] = annual_slope
            annual_CV_RMSE['Alltime'][Areas[iarea]][count] = annual_RMSE
            annual_CV_PWAModel['Alltime'][Areas[iarea]][count] = annual_PWA_model
            annual_CV_PWAMonitor['Alltime'][Areas[iarea]][count] = annual_PWA_monitor
            training_annual_CV_R2['Alltime'][Areas[iarea]][count] = training_annual_R2

            month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2_EachYear(test_index=area_test_index, final_data=overall_final_test[Areas[iarea]],
                                    test_obs_data=overall_obs_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                    beginyear=Area_beginyears[Areas[iarea]],
                                    endyear=endyear[-1])###Each Year 
            training_month_R2 = CalculateTrainingMonthR2(train_index=area_train_forStatistic_index,final_training_data=overall_train_final[Areas[iarea]],
                                                            train_obs_data=overall_train_obs[Areas[iarea]],
                                                            beginyear=beginyear[imodel],
                                                            endyear=endyear[imodel])
            training_month_CV_R2['Alltime'][Areas[iarea]][:, count] = training_month_R2
            month_CV_R2['Alltime'][Areas[iarea]][:, count] = month_R2
            month_CV_slope['Alltime'][Areas[iarea]][:, count] = month_slope
            month_CV_RMSE['Alltime'][Areas[iarea]][:, count] = month_RMSE
            month_CV_PWAModel['Alltime'][Areas[iarea]][:, count] = month_PWA_model
            month_CV_PWAMonitor['Alltime'][Areas[iarea]][:, count] = month_PWA_monitor

        count += 1
    txt_outdir = txt_dir + '{}/Results/results-SpatialCV/'.format(version)
    if not os.path.isdir(txt_outdir):
        os.makedirs(txt_outdir)
    txtoutfile = txt_outdir + 'Spatial_CV_'+ typeName +'_v' + version + '_' + str(nchannel) + 'Channel_' + str(width) + 'x' + str(width) + special_name + '.csv'

    for imodel in range(len(beginyear)):
        if imodel == 0:
            status = 'w'
        else:
            status = 'a'
        for iarea in range(len(MultiyearForMultiAreasList[imodel])):
            Optimal_Model_Output_Text(outfile=txtoutfile,status=status,training_annual_CV_R2=training_annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]]
                                      ,training_month_CV_R2=training_month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                                      CV_R2=CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_R2=annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_R2=month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_slope=CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_slope=annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_slope=month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_RMSE=CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_RMSE=annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_RMSE=month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_models=annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_monitors=annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_models=month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_monitors=month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    beginyear=beginyear[imodel],endyear=endyear[imodel],Area=MultiyearForMultiAreasList[imodel][iarea],
                    kfold=kfold,repeats=repeats)
            #regression_plot(plot_obs_pm25=annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],plot_pre_pm25=annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
            #            version=version,channel=nchannel,special_name=special_name,area_name=Area,beginyear=str(beginyear[imodel]),
            #            endyear=str(endyear[imodel]),extentlim=4.2*np.mean(annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]]),
            #            bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
            #             Log_PM25=Log_PM25)
    for iarea in range(len(Areas)):
        Optimal_Model_Output_Text(outfile=txtoutfile, status='a',training_annual_CV_R2=training_annual_CV_R2['Alltime'][Areas[iarea]]
                                   ,training_month_CV_R2=training_month_CV_R2['Alltime'][Areas[iarea]], CV_R2=CV_R2['Alltime'][Areas[iarea]],
                annual_CV_R2=annual_CV_R2['Alltime'][Areas[iarea]],
                month_CV_R2=month_CV_R2['Alltime'][Areas[iarea]],
                CV_slope=CV_slope['Alltime'][Areas[iarea]],
                annual_CV_slope=annual_CV_slope['Alltime'][Areas[iarea]],
                month_CV_slope=month_CV_slope['Alltime'][Areas[iarea]],
                CV_RMSE=CV_RMSE['Alltime'][Areas[iarea]],
                annual_CV_RMSE=annual_CV_RMSE['Alltime'][Areas[iarea]],
                month_CV_RMSE=month_CV_RMSE['Alltime'][Areas[iarea]], 
                annual_CV_models=annual_CV_PWAModel['Alltime'][Areas[iarea]], 
                annual_CV_monitors=annual_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                month_CV_models=month_CV_PWAModel['Alltime'][Areas[iarea]], 
                month_CV_monitors=month_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                beginyear='Alltime',
                endyear=' ',Area=Areas[iarea],
                kfold=kfold, repeats=repeats)
        regression_plot(plot_obs_pm25=annual_obs_dic['Alltime'][Areas[iarea]],
                    plot_pre_pm25=annual_final_dic['Alltime'][Areas[iarea]],
                    version=version, channel=nchannel, special_name=special_name, area_name=Areas[iarea],
                    beginyear='Alltime',
                    endyear='', extentlim=2.2 * np.mean(annual_obs_dic['Alltime'][Areas[iarea]]),
                     bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)
    del final_data, overall_final_test, obs_data, overall_obs_test,train_input, true_input
    gc.collect()

    return txtoutfile

def MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25_AllfoldsTogether_GBDAreas(train_input, true_input,channel_index, kfold:int, repeats:int,
                         extent,num_epochs:int, batch_size:int, learning_rate:float,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array,bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,EachMonthSlopeUnity:bool,
                         EachAreaForcedSlopeUnity:bool,
                         Log_PM25:bool):

    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    site_index = np.array(range(10870))         ### The index of sites.
    nchannel   = len(channel_index)    ### The number of channels.
    width      = train_input.shape[2]    ### The width of the input images.
    count      = 0                       ### Initialize the count number.
    seed = Get_CV_seed()                 ### Get the seed for random numbers for the folds seperation.
    ### Get observation data and Normalized parameters
    obs_data, obs_mean, obs_std = Get_data_NormPara(input_dir='/my-projects/Projects/MLCNN_PM25_2021/data/',input_file='obsPM25.npy')
    geo_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/geoPM25.npy')
    population_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/CoMonitors_Population_Data.npy')
    ### Initialize the CV R2 arrays for all datasets
    extent_dic = extent_table()
    #MultiyearForMultiAreasList = [['NA'],['NA'],['NA','EU'],['NA','EU','AS','GL']]## Each model test on which areas
    #Area_beginyears = {'NA':2001,'EU':2010,'AS':2015,'GL':2015}

    MultiyearForMultiAreasList = [['High_income_North_America'],['High_income_North_America'],
                                  ['Western_Europe','High_income_North_America','Eastern_Europe'],
                                  ['East_Asia','High_income_North_America','Eastern_Europe',
                                    'South_Asia','Western_Europe','Tropical_Latin_America',
                                    'Southeast_Asia','Rest_of_world','GL']]## Each model test on which areas

    Area_beginyears = {'High_income_North_America':2001,'Western_Europe':2010,
                       'East_Asia':2015,'South_Asia':2015,'Tropical_Latin_America':2015,'Eastern_Europe':2010,
                       'Southeast_Asia':2015,'Rest_of_world':2015,'GL':2015}
    Areas =['East_Asia','High_income_North_America','Eastern_Europe',
    'South_Asia','Western_Europe','Tropical_Latin_America',
    'Southeast_Asia','Rest_of_world','GL']
    
    beginyear = [2001,2005,2010,2015]
    #Areas = ['NA','EU','AS','GL']## Alltime areas names.
    training_CV_R2, training_annual_CV_R2,training_month_CV_R2, CV_R2, annual_CV_R2, month_CV_R2, CV_slope, annual_CV_slope, month_CV_slope, CV_RMSE, annual_CV_RMSE, month_CV_RMSE,annual_CV_PWAModel,month_CV_PWAModel,annual_CV_PWAMonitor,month_CV_PWAMonitor = Initialize_multiareas_optimalModel_CV_Dic(kfold=kfold,repeats=repeats,breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    annual_final_dic, annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
    training_annual_final_dic, training_annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)

    train_input,train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index)
    
    GeoPM25_mean = train_mean[16,int((width-1)/2),int((width-1)/2)]
    GeoPM25_std  = train_std[16,int((width-1)/2),int((width-1)/2)]
    SitesNumber_mean = train_mean[31,int((width-1)/2),int((width-1)/2)]
    SitesNumber_std  = train_std[31,int((width-1)/2),int((width-1)/2)]
    train_input = train_input[:,channel_index,:,:]

    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    
    for train_index, test_index in rkf.split(site_index):
        # *------------------------------------------------------------------------------*#
        ## Initialize the results arraies.
        ## For recording all models results in this fold.
        # *------------------------------------------------------------------------------*#
        overall_final_test = {}
        overall_geo_test   = {}
        overall_obs_test   = {}
        overall_train_final = {}
        overall_train_obs = {}
        overall_population_test = {}

        for iarea in Areas:
            overall_final_test[iarea] = np.array([],dtype = np.float64)
            overall_geo_test[iarea] = np.array([],dtype = np.float64)
            overall_obs_test[iarea] = np.array([],dtype = np.float64)
            overall_train_final[iarea] = np.array([],dtype = np.float64)
            overall_train_obs[iarea] = np.array([],dtype = np.float64)
            overall_population_test[iarea] = np.array([],dtype = np.float64)
        
        for imodel in range(len(beginyear)):

            X_index = GetTrainingIndex(Global_index=site_index,train_index=train_index,beginyear=beginyear[imodel],
                                            endyear=endyear[imodel],databeginyear=databeginyear,GLsitesNum=len(site_index))
            print('Train Index length: ', len(train_index),'\n X_index length: ', len(X_index))
            #X_train = Normlize_Training_Datasets(train_input=train_input[X_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #y_train = Normlize_Training_Datasets(train_input=train_input[Y_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #X_test,obs_mean,obs_std = Normlize_Testing_Datasets(true_input=true_input[X_index])  # Area Normalize True
            #X_test = true_input[X_index]

            X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
            
            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            cnn_model = ResNet(nchannel=nchannel,block=BasicBlock,blocks_num=[1,1,1,1],num_classes=1,include_top=True,
            groups=1,width_per_group=width)
            #cnn_model = Net(nchannel=nchannel)
            #cnn_model.apply(initialize_weights_Xavier) # No need for Residual Net

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)

            train_loss, train_acc = train(cnn_model, X_train, X_test,batch_size,learning_rate, num_epochs,GeoPM25_mean=GeoPM25_mean,GeoPM25_std=GeoPM25_std,SitesNumber_mean=SitesNumber_mean,SitesNumber_std=SitesNumber_std) 
            
            X_train_aug = []
            X_test_aug = []
            # *------------------------------------------------------------------------------*#
            ## Save Model results.
            # *------------------------------------------------------------------------------*#
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            modelfile = model_outdir + 'CNN_PM25_Spatial_'+typeName+'_'+Area+'_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
            torch.save(cnn_model, modelfile)

            #if ForcedUnitySlope == True:
            #    Training_Estimation = predict(X_train,cnn_model,width,3000)
            #    if bias == True:
            #        train_final_data = Training_Estimation + geo_data[X_index]
            #    elif Normlized_PM25 == True:
            #        train_final_data = Training_Estimation * obs_std + obs_mean
            #    elif Absolute_Pm25 == True:
            #        train_final_data = Training_Estimation
            #    elif Log_PM25 == True:
            #        train_final_data = np.exp(Training_Estimation) - 1
            #    regression_Dic = regress2(_x=X_test,_y=train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            #    offset,slope = regression_Dic['intercept'], regression_Dic['slope']                
            for iarea in range(len(MultiyearForMultiAreasList[imodel])):
                #extent = extent_dic[MultiyearForMultiAreasList[imodel][iarea]]
                #area_test_index = get_area_index(extent=extent, test_index=test_index)
                GBD_area_index = load_GBD_area_index(area=MultiyearForMultiAreasList[imodel][iarea])
                area_test_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=test_index)
                Y_index = GetValidationIndex(area_index=area_test_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]

                
                if EachAreaForcedSlopeUnity:
                    #area_train_index = get_area_index(extent=extent, test_index=train_index)
                    #area_train_forStatistic_index = get_area_index(extent=extent, test_index=train_index)
                    area_train_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                    area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                else:
                    #area_train_index = get_area_index(extent=extent_dic['GL'], test_index=train_index)
                    #area_train_forStatistic_index = get_area_index(extent=extent, test_index=train_index)
                    GL_GBD_area_index = load_GBD_area_index(area='GL')
                    area_train_index = get_test_index_inGBD_area(GBD_area_index=GL_GBD_area_index,test_index=train_index)
                    area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                
                XforForcedSlope_index = GetValidationIndex(area_index=area_train_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                XforStatistic_index = GetValidationIndex(area_index=area_train_forStatistic_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                x_train_forSlope = train_input[XforForcedSlope_index,:,:,:]
                X_train_forStatistic = train_input[XforStatistic_index,:,:,:]


                # *------------------------------------------------------------------------------*#
                ## Validation Process
                # *------------------------------------------------------------------------------*#
                Training_Prediction = predict(x_train_forSlope,cnn_model,width,3000)
                Training_forStatistic = predict(X_train_forStatistic, cnn_model, width,3000)
                Validation_Prediction = predict(y_train, cnn_model, width, 3000)
                if bias == True:
                    final_data = Validation_Prediction + geo_data[Y_index]
                    train_final_data = Training_Prediction + geo_data[XforForcedSlope_index]
                    train_final_forStatistic = Training_forStatistic + geo_data[XforStatistic_index]
                elif Normlized_PM25 == True:
                    final_data = Validation_Prediction * obs_std + obs_mean
                    train_final_data = Training_Prediction * obs_std + obs_mean
                    train_final_forStatistic = Training_forStatistic* obs_std + obs_mean
                elif Absolute_Pm25 == True:
                    final_data = Validation_Prediction
                    train_final_data = Training_Prediction
                    train_final_forStatistic = Training_forStatistic
                elif Log_PM25 == True:
                    final_data = np.exp(Validation_Prediction) - 1
                    train_final_data = np.exp(Training_Prediction) - 1
                    train_final_forStatistic = np.exp(train_final_forStatistic) - 1
                nearest_distance = get_nearest_test_distance(area_test_index=area_test_index,area_train_index=train_index)
                coeficient = get_coefficients(nearest_site_distance=nearest_distance)
                final_data = (1.0-coeficient)*final_data + coeficient * geo_data[Y_index]
                print('Forced Slope Unity - length of area_test_index: ',len(area_test_index),' length of area_train_index',len(area_train_index),'Area: ',MultiyearForMultiAreasList[imodel][iarea],
                      '\nlength of final_data',len(final_data))
                final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=obs_data[XforForcedSlope_index],
                                              test_final_data=final_data,train_area_index=area_train_index,test_area_index=area_test_index,endyear=endyear[imodel],
                                              beginyear=beginyear[imodel],EachMonth=EachMonthSlopeUnity)
                
                
                
                # *------------------------------------------------------------------------------*#
                ## Recording Results
                # *------------------------------------------------------------------------------*#
                test_obs_data = obs_data[Y_index]
                Train_obs_data = obs_data[XforStatistic_index]
                Validation_population = population_data[Y_index]
                overall_final_test[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_final_test[MultiyearForMultiAreasList[imodel][iarea]],final_data)
                overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]],test_obs_data)
                overall_population_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_population_test[MultiyearForMultiAreasList[imodel][iarea]],Validation_population)
                overall_train_final[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_train_final[MultiyearForMultiAreasList[imodel][iarea]],train_final_forStatistic)
                overall_train_obs[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_train_obs[MultiyearForMultiAreasList[imodel][iarea]], Train_obs_data)
                
                # *------------------------------------------------------------------------------*#
                ## Calculate the correlation R2 for this model this fold
                # *------------------------------------------------------------------------------*#
                print('Area: ',MultiyearForMultiAreasList[imodel][iarea],' fold:', str(count), ' beginyear: ', str(beginyear[imodel]),' endyear: ', str(endyear[imodel]))
                #
                #CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = linear_regression(final_data,test_obs_data)
                
                #CV_regression_Dic = regress2(_x=test_obs_data,_y=final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                #CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = CV_regression_Dic['slope']
                #CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = Cal_RMSE(test_obs_data,final_data)

                
                training_annual_R2 = CalculateTrainingAnnualR2(train_index=area_train_forStatistic_index,train_final_data=train_final_forStatistic,train_obs_data=Train_obs_data,
                                                               beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                #annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_final_dic[str(beginyear[imodel])],annual_final_data)
                #annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_obs_dic[str(beginyear[imodel])],annual_mean_obs)
                #annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_R2
                #annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_slope
                #annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_RMSE
                #annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_model
                #annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_monitor
                training_annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = training_annual_R2
                
                training_month_R2 = CalculateTrainingMonthR2(train_index=area_train_forStatistic_index,final_training_data=train_final_forStatistic,
                                                             train_obs_data=Train_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                
        # *------------------------------------------------------------------------------*#
        ## Calculate the correlation R2 for all models this fold
        # *------------------------------------------------------------------------------*#
        
        for iarea in range(len(Areas)):
            #extent = extent_dic[Areas[iarea]]
            GBD_area_index = load_GBD_area_index(area=Areas[iarea])
            area_test_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index, test_index=test_index)
            area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index, test_index=train_index)
            print('Area: ',Areas[iarea], ' fold: ',str(count),  ' - Alltime')
            #CV_R2['Alltime'][Areas[iarea]][count] = linear_regression(overall_final_test[Areas[iarea]], overall_obs_test[Areas[iarea]])
            #CV_regression_Dic = regress2(_x=overall_obs_test[Areas[iarea]],_y=overall_final_test[Areas[iarea]],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            #CV_slope['Alltime'][Areas[iarea]][count] = CV_regression_Dic['slope']
            #CV_RMSE['Alltime'][Areas[iarea]][count] = Cal_RMSE(overall_obs_test[Areas[iarea]],overall_final_test[Areas[iarea]])

            
            annual_final_data, annual_mean_obs = derive_Annual_data(test_index=area_test_index, 
                                                                        final_data=overall_final_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                                                          test_obs_data=overall_obs_test[Areas[iarea]],
                                                                          beginyear=Area_beginyears[Areas[iarea]],
                                                                          endyear=endyear[-1])
            
            training_annual_R2 = CalculateTrainingAnnualR2(train_index=area_train_forStatistic_index,train_final_data=overall_train_final[Areas[iarea]],
                                                          train_obs_data=overall_train_obs[Areas[iarea]],
                                                            beginyear=Area_beginyears[Areas[iarea]],
                                                           endyear=endyear[-1])
            annual_final_dic['Alltime'][Areas[iarea]] = np.append(annual_final_dic['Alltime'][Areas[iarea]], annual_final_data)
            annual_obs_dic['Alltime'][Areas[iarea]] = np.append(annual_obs_dic['Alltime'][Areas[iarea]],annual_mean_obs)
            
            #annual_CV_slope['Alltime'][Areas[iarea]][count] = annual_slope
            ##annual_CV_RMSE['Alltime'][Areas[iarea]][count] = annual_RMSE
            #annual_CV_PWAModel['Alltime'][Areas[iarea]][count] = annual_PWA_model
            #annual_CV_PWAMonitor['Alltime'][Areas[iarea]][count] = annual_PWA_monitor
            training_annual_CV_R2['Alltime'][Areas[iarea]][count] = training_annual_R2

            

        count += 1
    txt_outdir = txt_dir + '{}/Results/results-SpatialCV/'.format(version)
    if not os.path.isdir(txt_outdir):
        os.makedirs(txt_outdir)
    txtoutfile = txt_outdir + 'Spatial_CV_'+ typeName +'_v' + version + '_' + str(nchannel) + 'Channel_' + str(width) + 'x' + str(width) + special_name + '.csv'
    for iarea in range(len(Areas)):
        if imodel == 0:
            status = 'w'
        else:
            status = 'a'
        annual_CV_R2 = linear_regression(annual_final_dic['Alltime'][Areas[iarea]],annual_obs_dic['Alltime'][Areas[iarea]])
        Optimal_Model_GBDAreaAllfolds_Output_Text(outfile=txtoutfile, status=status,training_annual_CV_R2=training_annual_CV_R2['Alltime'][Areas[iarea]]
                                   , annual_CV_R2=annual_CV_R2,
               
                beginyear='Alltime',
                endyear=' ',Area=Areas[iarea],
                kfold=kfold, repeats=repeats)
        regression_plot(plot_obs_pm25=annual_obs_dic['Alltime'][Areas[iarea]],
                    plot_pre_pm25=annual_final_dic['Alltime'][Areas[iarea]],
                    version=version, channel=nchannel, special_name=special_name, area_name=Areas[iarea],
                    beginyear='Alltime',
                    endyear='', extentlim=2.2 * np.mean(annual_obs_dic['Alltime'][Areas[iarea]]),
                     bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)
    del final_data, overall_final_test, obs_data, overall_obs_test,train_input, true_input
    gc.collect()

    return txtoutfile


def MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25_GBDAreas(train_input, true_input,channel_index, kfold:int, repeats:int,
                         extent,num_epochs:int, batch_size:int, learning_rate:float,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array,bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,EachMonthSlopeUnity:bool,
                         EachAreaForcedSlopeUnity:bool,
                         Log_PM25:bool):

    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    site_index = np.array(range(10870))         ### The index of sites.
    nchannel   = len(channel_index)    ### The number of channels.
    width      = train_input.shape[2]    ### The width of the input images.
    count      = 0                       ### Initialize the count number.
    seed = Get_CV_seed()                 ### Get the seed for random numbers for the folds seperation.
    ### Get observation data and Normalized parameters
    obs_data, obs_mean, obs_std = Get_data_NormPara(input_dir='/my-projects/Projects/MLCNN_PM25_2021/data/',input_file='obsPM25.npy')
    geo_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/geoPM25.npy')
    population_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/CoMonitors_Population_Data.npy')
    ### Initialize the CV R2 arrays for all datasets
    extent_dic = extent_table()
    #MultiyearForMultiAreasList = [['NA'],['NA'],['NA','EU'],['NA','EU','AS','GL']]## Each model test on which areas
    #Area_beginyears = {'NA':2001,'EU':2010,'AS':2015,'GL':2015}

    MultiyearForMultiAreasList = [['High_income_North_America'],['High_income_North_America'],
                                  ['Eastern_Europe','Western_Europe','High_income_North_America'],
                                  ['East_Asia','Eastern_Europe','High_income_North_America',
                                   'South_Asia','Western_Europe','Tropical_Latin_America',
                                    'Southeast_Asia','Rest_of_world','GL']]## Each model test on which areas

    Area_beginyears = {'High_income_North_America':2001,'Eastern_Europe':2010,'Western_Europe':2010,
                       'East_Asia':2015,'South_Asia':2015,'Tropical_Latin_America':2015,
                       'Southeast_Asia':2015,'Rest_of_world':2015,'GL':2015}
    
    Areas =['East_Asia','Eastern_Europe','High_income_North_America',
    'South_Asia','Western_Europe','Tropical_Latin_America',
    'Southeast_Asia','Rest_of_world','GL']

    #MultiyearForMultiAreasList = [
    #                              ['East_Asia',
    #                                'South_Asia','Tropical_Latin_America',
    #                                'Southeast_Asia','Rest_of_world','GL']]## Each model test on which areas

    #Area_beginyears = {
    #                   'East_Asia':2015,'South_Asia':2015,'Tropical_Latin_America':2015,
    #                   'Southeast_Asia':2015,'Rest_of_world':2015,'GL':2015}
    #Areas =['East_Asia',
    #'South_Asia','Tropical_Latin_America',
    #'Southeast_Asia','Rest_of_world','GL']
    
    beginyear = [2015]
    #Areas = ['NA','EU','AS','GL']## Alltime areas names.
    training_CV_R2, training_annual_CV_R2,training_month_CV_R2, CV_R2, annual_CV_R2, month_CV_R2, CV_slope, annual_CV_slope, month_CV_slope, CV_RMSE, annual_CV_RMSE, month_CV_RMSE,annual_CV_PWAModel,month_CV_PWAModel,annual_CV_PWAMonitor,month_CV_PWAMonitor = Initialize_multiareas_optimalModel_CV_Dic(kfold=kfold,repeats=repeats,breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    annual_final_dic, annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)
    training_annual_final_dic, training_annual_obs_dic = Initialize_DataRecording_MultiAreas_Dic(breakpoints=beginyear,MultiyearsForAreas=MultiyearForMultiAreasList)

    train_input,train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index)
    
    GeoPM25_mean = train_mean[16,int((width-1)/2),int((width-1)/2)]
    GeoPM25_std  = train_std[16,int((width-1)/2),int((width-1)/2)]
    SitesNumber_mean = train_mean[31,int((width-1)/2),int((width-1)/2)]
    SitesNumber_std  = train_std[31,int((width-1)/2),int((width-1)/2)]
    train_input = train_input[:,channel_index,:,:]

    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    Allfolds_annual_final_test = {}
    Allfolds_annual_obs_test   = {}
    Allfolds_monthly_final_test   = {}
    Allfolds_monthly_obs_test = {}
    

    for iarea in Areas:
        Allfolds_annual_final_test[iarea] = np.array([],dtype = np.float64)
        Allfolds_annual_obs_test[iarea] = np.array([],dtype = np.float64)
        
    for train_index, test_index in rkf.split(site_index):
        # *------------------------------------------------------------------------------*#
        ## Initialize the results arraies.
        ## For recording all models results in this fold.
        # *------------------------------------------------------------------------------*#
        overall_final_test = {}
        overall_geo_test   = {}
        overall_obs_test   = {}
        overall_train_final = {}
        overall_train_obs = {}
        overall_population_test = {}

        for iarea in Areas:
            overall_final_test[iarea] = np.array([],dtype = np.float64)
            overall_geo_test[iarea] = np.array([],dtype = np.float64)
            overall_obs_test[iarea] = np.array([],dtype = np.float64)
            overall_train_final[iarea] = np.array([],dtype = np.float64)
            overall_train_obs[iarea] = np.array([],dtype = np.float64)
            overall_population_test[iarea] = np.array([],dtype = np.float64)
        
        for imodel in range(len(beginyear)):

            X_index = GetTrainingIndex(Global_index=site_index,train_index=train_index,beginyear=beginyear[imodel],
                                            endyear=endyear[imodel],databeginyear=databeginyear,GLsitesNum=len(site_index))
            print('Train Index length: ', len(train_index),'\n X_index length: ', len(X_index))
            #X_train = Normlize_Training_Datasets(train_input=train_input[X_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #y_train = Normlize_Training_Datasets(train_input=train_input[Y_index,:,:,:],channel_index=channel_index) # Area Normlize Training Data
            #X_test,obs_mean,obs_std = Normlize_Testing_Datasets(true_input=true_input[X_index])  # Area Normalize True
            #X_test = true_input[X_index]

            X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
            
            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            cnn_model = ResNet(nchannel=nchannel,block=BasicBlock,blocks_num=[1,1,1,1],num_classes=1,include_top=True,
            groups=1,width_per_group=width)
            #cnn_model = Net(nchannel=nchannel)
            #cnn_model.apply(initialize_weights_Xavier) # No need for Residual Net

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)

            train_loss, train_acc = train(cnn_model, X_train, X_test,batch_size,learning_rate, num_epochs,GeoPM25_mean=GeoPM25_mean,GeoPM25_std=GeoPM25_std,SitesNumber_mean=SitesNumber_mean,SitesNumber_std=SitesNumber_std) 
            X_train_aug = []
            X_test_aug = []
            # *------------------------------------------------------------------------------*#
            ## Save Model results.
            # *------------------------------------------------------------------------------*#
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            modelfile = model_outdir + 'CNN_PM25_Spatial_'+typeName+'_'+Area+'_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
            torch.save(cnn_model, modelfile)

            for iarea in range(len(MultiyearForMultiAreasList[imodel])):
                GBD_area_index = load_GBD_area_index(area=MultiyearForMultiAreasList[imodel][iarea])
                area_test_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=test_index)
                Y_index = GetValidationIndex(area_index=area_test_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]
                if EachAreaForcedSlopeUnity:
                    area_train_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                    area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                else:
                    GL_GBD_area_index = load_GBD_area_index(area='GL')
                    area_train_index = get_test_index_inGBD_area(GBD_area_index=GL_GBD_area_index,test_index=train_index)
                    area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index,test_index=train_index)
                XforForcedSlope_index = GetValidationIndex(area_index=area_train_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                XforStatistic_index = GetValidationIndex(area_index=area_train_forStatistic_index,beginyear=beginyear[imodel],endyear=endyear[imodel],GLsitesNum=len(site_index))
                x_train_forSlope = train_input[XforForcedSlope_index,:,:,:]
                X_train_forStatistic = train_input[XforStatistic_index,:,:,:]
                # *------------------------------------------------------------------------------*#
                ## Validation Process
                # *------------------------------------------------------------------------------*#
                Training_Prediction = predict(x_train_forSlope,cnn_model,width,3000)
                Training_forStatistic = predict(X_train_forStatistic, cnn_model, width,3000)
                Validation_Prediction = predict(y_train, cnn_model, width, 3000)
                if bias == True:
                    final_data = Validation_Prediction + geo_data[Y_index]
                    train_final_data = Training_Prediction + geo_data[XforForcedSlope_index]
                    train_final_forStatistic = Training_forStatistic + geo_data[XforStatistic_index]
                elif Normlized_PM25 == True:
                    final_data = Validation_Prediction * obs_std + obs_mean
                    train_final_data = Training_Prediction * obs_std + obs_mean
                    train_final_forStatistic = Training_forStatistic* obs_std + obs_mean
                elif Absolute_Pm25 == True:
                    final_data = Validation_Prediction
                    train_final_data = Training_Prediction
                    train_final_forStatistic = Training_forStatistic
                elif Log_PM25 == True:
                    final_data = np.exp(Validation_Prediction) - 1
                    train_final_data = np.exp(Training_Prediction) - 1
                    train_final_forStatistic = np.exp(train_final_forStatistic) - 1
                nearest_distance = get_nearest_test_distance(area_test_index=area_test_index,area_train_index=train_index)
                coeficient = get_coefficients(nearest_site_distance=nearest_distance,beginyear=beginyear[imodel],
                                              endyear = endyear[imodel])
                final_data = (1.0-coeficient)*final_data + coeficient * geo_data[Y_index]
                print('Forced Slope Unity - length of area_test_index: ',len(area_test_index),' length of area_train_index',len(area_train_index),'Area: ',MultiyearForMultiAreasList[imodel][iarea],
                      '\nlength of final_data',len(final_data))
                final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=obs_data[XforForcedSlope_index],
                                              test_final_data=final_data,train_area_index=area_train_index,test_area_index=area_test_index,endyear=endyear[imodel],
                                              beginyear=beginyear[imodel],EachMonth=EachMonthSlopeUnity)
                
                
                
                # *------------------------------------------------------------------------------*#
                ## Recording Results
                # *------------------------------------------------------------------------------*#
                test_obs_data = obs_data[Y_index]
                Train_obs_data = obs_data[XforStatistic_index]
                Validation_population = population_data[Y_index]
                overall_final_test[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_final_test[MultiyearForMultiAreasList[imodel][iarea]],final_data)
                overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_obs_test[MultiyearForMultiAreasList[imodel][iarea]],test_obs_data)
                overall_population_test[MultiyearForMultiAreasList[imodel][iarea]]   = np.append(overall_population_test[MultiyearForMultiAreasList[imodel][iarea]],Validation_population)
                overall_train_final[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_train_final[MultiyearForMultiAreasList[imodel][iarea]],train_final_forStatistic)
                overall_train_obs[MultiyearForMultiAreasList[imodel][iarea]] = np.append(overall_train_obs[MultiyearForMultiAreasList[imodel][iarea]], Train_obs_data)
                
                # *------------------------------------------------------------------------------*#
                ## Calculate the correlation R2 for this model this fold
                # *------------------------------------------------------------------------------*#
                print('Area: ',MultiyearForMultiAreasList[imodel][iarea],' fold:', str(count), ' beginyear: ', str(beginyear[imodel]),' endyear: ', str(endyear[imodel]))
                CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = linear_regression(final_data,test_obs_data)
                CV_regression_Dic = regress2(_x=test_obs_data,_y=final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = CV_regression_Dic['slope']
                CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = Cal_RMSE(test_obs_data,final_data)



                annual_R2,annual_final_data,annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2(test_index=area_test_index,final_data=final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                training_annual_R2 = CalculateTrainingAnnualR2(train_index=area_train_forStatistic_index,train_final_data=train_final_forStatistic,train_obs_data=Train_obs_data,
                                                               beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_final_dic[str(beginyear[imodel])],annual_final_data)
                annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]] = np.append(annual_obs_dic[str(beginyear[imodel])],annual_mean_obs)
                annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_R2
                annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_slope
                annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_RMSE
                annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_model
                annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = annual_PWA_monitor
                training_annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][count] = training_annual_R2
                
                training_month_R2 = CalculateTrainingMonthR2(train_index=area_train_forStatistic_index,final_training_data=train_final_forStatistic,
                                                             train_obs_data=Train_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2(test_index=area_test_index,final_data = final_data,population=Validation_population,
                                                                            test_obs_data=test_obs_data,
                                                                            beginyear=beginyear[imodel],
                                                                            endyear=endyear[imodel])
                month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_R2
                month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_slope
                month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_RMSE
                month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_model
                month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = month_PWA_monitor
                training_month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]][:,count] = training_month_R2



        # *------------------------------------------------------------------------------*#
        ## Calculate the correlation R2 for all models this fold
        # *------------------------------------------------------------------------------*#
        
        for iarea in range(len(Areas)):
            #extent = extent_dic[Areas[iarea]]
            GBD_area_index = load_GBD_area_index(area=Areas[iarea])
            area_test_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index, test_index=test_index)
            area_train_forStatistic_index = get_test_index_inGBD_area(GBD_area_index=GBD_area_index, test_index=train_index)
            print('Area: ',Areas[iarea], ' fold: ',str(count),  ' - Alltime')
            CV_R2['Alltime'][Areas[iarea]][count] = linear_regression(overall_final_test[Areas[iarea]], overall_obs_test[Areas[iarea]])
            CV_regression_Dic = regress2(_x=overall_obs_test[Areas[iarea]],_y=overall_final_test[Areas[iarea]],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            CV_slope['Alltime'][Areas[iarea]][count] = CV_regression_Dic['slope']
            CV_RMSE['Alltime'][Areas[iarea]][count] = Cal_RMSE(overall_obs_test[Areas[iarea]],overall_final_test[Areas[iarea]])

            
            annual_R2, annual_final_data, annual_mean_obs,annual_slope, annual_RMSE,annual_PWA_model,annual_PWA_monitor = CalculateAnnualR2(test_index=area_test_index, 
                                                                        final_data=overall_final_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                                                          test_obs_data=overall_obs_test[Areas[iarea]],
                                                                          beginyear=Area_beginyears[Areas[iarea]],
                                                                          endyear=endyear[-1])
            
            training_annual_R2 = CalculateTrainingAnnualR2(train_index=area_train_forStatistic_index,train_final_data=overall_train_final[Areas[iarea]],
                                                          train_obs_data=overall_train_obs[Areas[iarea]],
                                                            beginyear=Area_beginyears[Areas[iarea]],
                                                           endyear=endyear[-1])
            annual_final_dic['Alltime'][Areas[iarea]] = np.append(annual_final_dic['Alltime'][Areas[iarea]], annual_final_data)
            annual_obs_dic['Alltime'][Areas[iarea]] = np.append(annual_obs_dic['Alltime'][Areas[iarea]],annual_mean_obs)
            annual_CV_R2['Alltime'][Areas[iarea]][count] = annual_R2
            annual_CV_slope['Alltime'][Areas[iarea]][count] = annual_slope
            annual_CV_RMSE['Alltime'][Areas[iarea]][count] = annual_RMSE
            annual_CV_PWAModel['Alltime'][Areas[iarea]][count] = annual_PWA_model
            annual_CV_PWAMonitor['Alltime'][Areas[iarea]][count] = annual_PWA_monitor
            training_annual_CV_R2['Alltime'][Areas[iarea]][count] = training_annual_R2

            month_R2,month_slope, month_RMSE,month_PWA_model, month_PWA_monitor = CalculateMonthR2(test_index=area_test_index, final_data=overall_final_test[Areas[iarea]],
                                    test_obs_data=overall_obs_test[Areas[iarea]],population=overall_population_test[Areas[iarea]],
                                    beginyear=Area_beginyears[Areas[iarea]],
                                    endyear=endyear[-1])
            training_month_R2 = CalculateTrainingMonthR2(train_index=area_train_forStatistic_index,final_training_data=overall_train_final[Areas[iarea]],
                                                            train_obs_data=overall_train_obs[Areas[iarea]],
                                                            beginyear=beginyear[imodel],
                                                            endyear=endyear[imodel])
            training_month_CV_R2['Alltime'][Areas[iarea]][:, count] = training_month_R2
            month_CV_R2['Alltime'][Areas[iarea]][:, count] = month_R2
            month_CV_slope['Alltime'][Areas[iarea]][:, count] = month_slope
            month_CV_RMSE['Alltime'][Areas[iarea]][:, count] = month_RMSE
            month_CV_PWAModel['Alltime'][Areas[iarea]][:, count] = month_PWA_model
            month_CV_PWAMonitor['Alltime'][Areas[iarea]][:, count] = month_PWA_monitor

        count += 1
    
    # *------------------------------------------------------------------------------*#
    ## Calculate the correlation R2 for all models for all folds
    # *------------------------------------------------------------------------------*#
    txt_outdir = txt_dir + '{}/Results/results-SpatialCV/'.format(version)
    if not os.path.isdir(txt_outdir):
        os.makedirs(txt_outdir)
    txtoutfile = txt_outdir + 'Spatial_CV_'+ typeName +'_v' + version + '_' + str(nchannel) + 'Channel_' + str(width) + 'x' + str(width) + special_name + '.csv'

    for imodel in range(len(beginyear)):
        if imodel == 0:
            status = 'w'
        else:
            status = 'a'
        for iarea in range(len(MultiyearForMultiAreasList[imodel])):
            Optimal_Model_Output_Text(outfile=txtoutfile,status=status,training_annual_CV_R2=training_annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]]
                                      ,training_month_CV_R2=training_month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                                      CV_R2=CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_R2=annual_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_R2=month_CV_R2[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_slope=CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_slope=annual_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_slope=month_CV_slope[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    CV_RMSE=CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_RMSE=annual_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_RMSE=month_CV_RMSE[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_models=annual_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    annual_CV_monitors=annual_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_models=month_CV_PWAModel[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    month_CV_monitors=month_CV_PWAMonitor[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
                    beginyear=beginyear[imodel],endyear=endyear[imodel],Area=MultiyearForMultiAreasList[imodel][iarea],
                    kfold=kfold,repeats=repeats)
            #regression_plot(plot_obs_pm25=annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],plot_pre_pm25=annual_final_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]],
            #            version=version,channel=nchannel,special_name=special_name,area_name=Area,beginyear=str(beginyear[imodel]),
            #            endyear=str(endyear[imodel]),extentlim=4.2*np.mean(annual_obs_dic[str(beginyear[imodel])][MultiyearForMultiAreasList[imodel][iarea]]),
            #            bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
            #             Log_PM25=Log_PM25)
    for iarea in range(len(Areas)):
        Optimal_Model_Output_Text(outfile=txtoutfile, status='a',training_annual_CV_R2=training_annual_CV_R2['Alltime'][Areas[iarea]]
                                   ,training_month_CV_R2=training_month_CV_R2['Alltime'][Areas[iarea]], CV_R2=CV_R2['Alltime'][Areas[iarea]],
                annual_CV_R2=annual_CV_R2['Alltime'][Areas[iarea]],
                month_CV_R2=month_CV_R2['Alltime'][Areas[iarea]],
                CV_slope=CV_slope['Alltime'][Areas[iarea]],
                annual_CV_slope=annual_CV_slope['Alltime'][Areas[iarea]],
                month_CV_slope=month_CV_slope['Alltime'][Areas[iarea]],
                CV_RMSE=CV_RMSE['Alltime'][Areas[iarea]],
                annual_CV_RMSE=annual_CV_RMSE['Alltime'][Areas[iarea]],
                month_CV_RMSE=month_CV_RMSE['Alltime'][Areas[iarea]], 
                annual_CV_models=annual_CV_PWAModel['Alltime'][Areas[iarea]], 
                annual_CV_monitors=annual_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                month_CV_models=month_CV_PWAModel['Alltime'][Areas[iarea]], 
                month_CV_monitors=month_CV_PWAMonitor['Alltime'][Areas[iarea]], 
                beginyear='Alltime',
                endyear=' ',Area=Areas[iarea],
                kfold=kfold, repeats=repeats)
        regression_plot(plot_obs_pm25=annual_obs_dic['Alltime'][Areas[iarea]],
                    plot_pre_pm25=annual_final_dic['Alltime'][Areas[iarea]],
                    version=version, channel=nchannel, special_name=special_name, area_name=Areas[iarea],
                    beginyear='Alltime',
                    endyear='', extentlim=2.2 * np.mean(annual_obs_dic['Alltime'][Areas[iarea]]),
                     bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)
    del final_data, overall_final_test, obs_data, overall_obs_test,train_input, true_input
    gc.collect()

    return txtoutfile


def MultiyearMultiAreas_AVD_SpatialCrossValidation_CombineWithGeophysicalPM25(train_input, true_input,channel_index, kfold:int, repeats:int,
                         extent,num_epochs:int, batch_size:int, learning_rate:float,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array,bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,EachMonthSlopeUnity:bool,
                         Log_PM25:bool):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    site_index = np.array(range(10870))         ### The index of sites.
    nchannel   = len(channel_index)    ### The number of channels.
    width      = train_input.shape[2]    ### The width of the input images.
    count      = 0                       ### Initialize the count number.
    seed = Get_CV_seed()                 ### Get the seed for random numbers for the folds seperation.
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ### Get observation data and Normalized parameters
    obs_data, obs_mean, obs_std = Get_data_NormPara(input_dir='/my-projects/Projects/MLCNN_PM25_2021/data/',input_file='obsPM25.npy')
    geo_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/geoPM25.npy')
    population_data = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/CoMonitors_Population_Data.npy')
    ### Initialize the CV R2 arrays for all datasets
    extent_dic = extent_table()
    #MultiyearForMultiAreasList = [['NA'],['NA'],['NA','EU'],['NA','EU','AS','GL']]## Each model test on which areas
    #Area_beginyears = {'NA':2001,'EU':2010,'AS':2015,'GL':2015}
    MultiyearForMultiAreasList = MultiyearForMultiAreasLists ## Each model test on which areas
    Area_beginyears = {'NA':NA_beginyear,'EU':EU_beginyear,'AS':AS_beginyear,'GL':GL_beginyear}
    Areas = ['NA','EU','AS','GL']## Alltime areas names.
    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(Areas=Areas,beginyear=beginyear[0],endyear=endyears[-1])
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    train_input,train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index)
    GeoPM25_mean = train_mean[16,int((width-1)/2),int((width-1)/2)]
    GeoPM25_std  = train_std[16,int((width-1)/2),int((width-1)/2)]
    SitesNumber_mean = train_mean[31,int((width-1)/2),int((width-1)/2)]
    SitesNumber_std  = train_std[31,int((width-1)/2),int((width-1)/2)]
    train_input = train_input[:,channel_index,:,:]
    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    count = 0
    for train_index, test_index in rkf.split(site_index):
        for imodel in range(len(beginyear)):
            X_index = GetTrainingIndex(Global_index=site_index,train_index=train_index,beginyear=beginyear[imodel],
                                            endyear=endyear[imodel],databeginyear=databeginyear,GLsitesNum=len(site_index))
            X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
            cnn_model = ResNet(nchannel=nchannel,block=BasicBlock,blocks_num=[1,1,1,1],num_classes=1,include_top=True,
            groups=1,width_per_group=width)
            #cnn_model = Net(nchannel=nchannel)
            #cnn_model.apply(initialize_weights_Xavier) # No need for Residual Net

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            torch.manual_seed(21)

            train_loss, train_acc = train(cnn_model, X_train, X_test,batch_size,learning_rate, num_epochs,GeoPM25_mean=GeoPM25_mean,GeoPM25_std=GeoPM25_std,SitesNumber_mean=SitesNumber_mean,SitesNumber_std=SitesNumber_std) 
           
            # *------------------------------------------------------------------------------*#
            ## Save Model results.
            # *------------------------------------------------------------------------------*#
           
            if not os.path.isdir(model_outdir):
                os.makedirs(model_outdir)
            modelfile = model_outdir + 'CNN_PM25_Spatial_'+typeName+'_'+Area+'_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
            torch.save(cnn_model, modelfile)
            print('iModel: {}, fold: {}'.format(imodel, count))
            for iyear in range((endyear[imodel]-beginyear[imodel]+1)):
                for iarea in range(len(MultiyearForMultiAreasList[imodel])):
                    
                    extent = extent_dic[MultiyearForMultiAreasList[imodel][iarea]]
                    area_test_index = get_area_index(extent=extent, test_index=test_index)
                    Y_index = GetValidationIndex(area_index=area_test_index,beginyear=(beginyear[imodel]+iyear),endyear=(beginyear[imodel]+iyear),GLsitesNum=len(site_index))
                    y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]
                    
                    area_train_forSlope_index = get_area_index(extent=extent_dic['GL'], test_index=train_index)
                    area_train_forStatistic_index = get_area_index(extent=extent, test_index=train_index)

                    XforForcedSlope_index = GetValidationIndex(area_index=area_train_forSlope_index,beginyear=(beginyear[imodel]+iyear),endyear=(beginyear[imodel]+iyear),GLsitesNum=len(site_index))
                    XforStatistic_index = GetValidationIndex(area_index=area_train_forStatistic_index,beginyear=(beginyear[imodel]+iyear),endyear=(beginyear[imodel]+iyear),GLsitesNum=len(site_index))
                    x_train_forSlope = train_input[XforForcedSlope_index,:,:,:]
                    X_train_forStatistic = train_input[XforStatistic_index,:,:,:]
                    # *------------------------------------------------------------------------------*#
                    ## Validation Process
                    # *------------------------------------------------------------------------------*#

                    Training_Prediction = predict(x_train_forSlope,cnn_model,width,3000)
                    Training_forStatistic = predict(X_train_forStatistic, cnn_model, width,3000)
                    Validation_Prediction = predict(y_train, cnn_model, width, 3000)
                    if bias == True:
                        final_data = Validation_Prediction + geo_data[Y_index]
                        train_final_data = Training_Prediction + geo_data[XforForcedSlope_index]
                        train_final_forStatistic = Training_forStatistic + geo_data[XforStatistic_index]
                    elif Normlized_PM25 == True:
                        final_data = Validation_Prediction * obs_std + obs_mean
                        train_final_data = Training_Prediction * obs_std + obs_mean
                        train_final_forStatistic = Training_forStatistic* obs_std + obs_mean
                    elif Absolute_Pm25 == True:
                        final_data = Validation_Prediction
                        train_final_data = Training_Prediction
                        train_final_forStatistic = Training_forStatistic
                    elif Log_PM25 == True:
                        final_data = np.exp(Validation_Prediction) - 1
                        train_final_data = np.exp(Training_Prediction) - 1
                        train_final_forStatistic = np.exp(train_final_forStatistic) - 1
                    nearest_distance = get_nearest_test_distance(area_test_index=area_test_index,area_train_index=train_index)
                    coeficient = get_coefficients(nearest_site_distance=nearest_distance,beginyear=(beginyear[imodel]+iyear),
                                              endyear = (beginyear[imodel]+iyear))
                    final_data = (1.0-coeficient)*final_data + coeficient * geo_data[Y_index]
                    print('Forced Slope Unity - length of area_test_index: ',len(area_test_index),' length of area_train_index',len(area_train_forSlope_index),'Area: ',MultiyearForMultiAreasList[imodel][iarea],
                      '\nlength of final_data',len(final_data))
                    final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=obs_data[XforForcedSlope_index],
                                              test_final_data=final_data,train_area_index=area_train_forSlope_index,test_area_index=area_test_index,endyear=(beginyear[imodel]+iyear),
                                              beginyear=(beginyear[imodel]+iyear),EachMonth=EachMonthSlopeUnity)
                    # *------------------------------------------------------------------------------*#
                    ## Recording Results
                    # *------------------------------------------------------------------------------*#
                    test_geo_data = geo_data[Y_index]
                    test_obs_data = obs_data[Y_index]
                    Train_obs_data = obs_data[XforStatistic_index]
                    Validation_population = population_data[Y_index]
                    for imonth in range(len(MONTH)):
                        final_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]] = \
                            np.append(final_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]],
                                       final_data[imonth*len(area_test_index):(imonth+1)*len(area_test_index)])
                        
                        obs_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]] = \
                            np.append(obs_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]],
                                       test_obs_data[imonth*len(area_test_index):(imonth+1)*len(area_test_index)])
                        
                        geo_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]] = \
                            np.append(geo_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]],
                                       test_geo_data[imonth*len(area_test_index):(imonth+1)*len(area_test_index)])
                        
                        testing_population_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]] = \
                            np.append(testing_population_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]],
                                       Validation_population[imonth*len(area_test_index):(imonth+1)*len(area_test_index)])
                        
                        training_final_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]] = \
                            np.append(training_final_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]],
                                       train_final_forStatistic[imonth*len(area_train_forStatistic_index):(imonth+1)*len(area_train_forStatistic_index)])
                        
                        training_obs_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]] = \
                            np.append(training_obs_data_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]],
                                       Train_obs_data[imonth*len(area_train_forStatistic_index):(imonth+1)*len(area_train_forStatistic_index)])
                        
                        training_dataForSlope_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]] = \
                            np.append(training_dataForSlope_recording[MultiyearForMultiAreasList[imodel][iarea]][str(beginyear[imodel]+iyear)][MONTH[imonth]],
                                       train_final_data[imonth*len(area_train_forSlope_index):(imonth+1)*len(area_train_forSlope_index)])
        count += 1
    # *------------------------------------------------------------------------------*#
    ## Calculate R2, RMSE, slope, etc.
    # *------------------------------------------------------------------------------*#
    test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, slope_CV_R2, PWAModel, PWAMonitors = calculate_Statistics_results(Areas=Areas,Area_beginyears=Area_beginyears, endyear=endyears[-1], 
                                                                                                                                   final_data_recording=final_data_recording,obs_data_recording=obs_data_recording,
                                                                                                                                   geo_data_recording=geo_data_recording,
                                                                                                                                   testing_population_data_recording=testing_population_data_recording,
                                                                                                                                   training_final_data_recording=training_final_data_recording, 
                                                                                                                                   training_obs_data_recording=training_obs_data_recording,
                                                                                                                                   )
    txt_outdir = txt_dir + '{}/Results/results-SpatialCV/'.format(version)
    if not os.path.isdir(txt_outdir):
        os.makedirs(txt_outdir)
    txtoutfile = txt_outdir + 'Spatial_CV_'+ typeName +'_v' + version + '_' + str(nchannel) + 'Channel_' + str(width) + 'x' + str(width) + special_name + '.csv'

    output_text(outfile=txtoutfile,status='w', Areas=Areas, Area_beginyears=Area_beginyears, endyear=endyears[-1],test_CV_R2=test_CV_R2,train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2,
                RMSE_CV_R2=RMSE_CV_R2,slope_CV_R2=slope_CV_R2,PWAModel=PWAModel, PWAMonitors=PWAMonitors)
    
    save_loss_accuracy(model_outdir=model_outdir,loss=train_loss,accuracy=train_acc,typeName=typeName,epoch=epoch,nchannel=nchannel,special_name=special_name,width=width,height=width)
    Loss_Accuracy_outdirs = Loss_Accuracy_outdir + '{}/Figures/figures-Loss_Accuracy/'.format(version)
    if not os.path.isdir(Loss_Accuracy_outdirs):
        os.makedirs(Loss_Accuracy_outdirs)
    Loss_Accuracy_outfile = Loss_Accuracy_outdirs + 'SpatialCV_{}_{}_{}Epoch_{}Channel_{}x{}{}.png'.format(typeName,version,epoch,nchannel,width,width,special_name)
    plot_loss_accuracy_with_epoch(loss=train_loss, accuracy=train_acc, outfile=Loss_Accuracy_outfile)

    for iarea in Areas:
        longterm_final_data, longterm_obs_data = get_longterm_array(area=iarea,imonth='Annual', beginyear=Area_beginyears[iarea], endyear=endyear[-1], final_data_recording=final_data_recording,
                                                                    obs_data_recording=obs_data_recording)
        regression_plot(plot_obs_pm25=longterm_obs_data,plot_pre_pm25=longterm_final_data,version=version,channel=nchannel,special_name=special_name,area_name=iarea,beginyear=Area_beginyears[iarea],endyear=endyear[-1],
                        extentlim=2.2 * np.mean(longterm_obs_data), bias=bias,Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,Log_PM25=Log_PM25)
    return txtoutfile


def plot_from_data(infile:str,true_infile,
 Area:str,version:str,special_name:str,nchannel:int,bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,
                         Log_PM25:bool):
    site_index = np.array(range(10870))  
    
    
    model_results = np.load(infile)
    obsPM25 = np.load(true_infile)
    
    regression_plot(plot_obs_pm25=obsPM25,
                    plot_pre_pm25=model_results,
                    version=version, channel=nchannel, special_name=special_name, area_name=Area,
                    beginyear='Alltime',
                    endyear='', extentlim=2.2 * np.mean(obsPM25),
                     bias=bias, Normlized_PM25=Normlized_PM25, Absolute_Pm25=Absolute_Pm25,
                         Log_PM25=Log_PM25)
    
    return


def Output_Text(outfile:str,status:str,CV_R2,annual_CV_R2,month_CV_R2,CV_slope,annual_CV_slope,month_CV_slope,
                CV_RMSE,annual_CV_RMSE,month_CV_RMSE,annual_CV_models,annual_CV_monitors,month_CV_models,month_CV_monitors,beginyear:str,
                endyear:str,Area:str,kfold:int,repeats:int):
    MONTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    CV_R2[-1] = np.mean(CV_R2[0:kfold * repeats])
    annual_CV_R2[-1] = np.mean(annual_CV_R2[0:kfold * repeats])
    CV_slope[-1] = np.mean(CV_slope[0:kfold * repeats])
    annual_CV_slope[-1] = np.mean(annual_CV_slope[0:kfold * repeats])
    CV_RMSE[-1] = np.mean(CV_RMSE[0:kfold * repeats])
    annual_CV_RMSE[-1] = np.mean(annual_CV_RMSE[0:kfold * repeats])

    annual_CV_models[-1] = np.mean(annual_CV_models[0:kfold * repeats])
    annual_CV_monitors[-1] = np.mean(annual_CV_monitors[0:kfold * repeats])
   

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Area,'Area ','Time Period: ', beginyear,' ', endyear])
        writer.writerow(['R2 for monthly validation','\nMax: ',str(np.round(np.max(CV_R2),4)),'Min: ',str(np.round(np.min(CV_R2),4)),
                         'Avg: ',str(np.round(CV_R2[-1],4)),'\nSlope for monthly validation','Max: ',str(np.round(np.max(CV_slope),4)),'Min: ',str(np.round(np.min(CV_slope),4)),
                         'Avg: ',str(np.round(CV_slope[-1],4)),'\nRMSE for monthly validation','Max: ',str(np.round(np.max(CV_RMSE),4)),'Min: ',str(np.round(np.min(CV_RMSE),4)),
                         'Avg: ',str(np.round(CV_RMSE[-1],4))])
        writer.writerow(['#####################   Annual average validation ####################','\nR2 Max: ', str(np.round(np.max(annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_R2), 4)),
                         'Avg: ', str(np.round(annual_CV_R2[-1], 4)),' \nSlope for Annual average validation', 'Max: ', str(np.round(np.max(annual_CV_slope), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_slope), 4)),
                         'Avg: ', str(np.round(annual_CV_slope[-1], 4)),' \nRMSE for Annual average validation', 'Max: ', str(np.round(np.max(annual_CV_RMSE), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_RMSE), 4)),
                         'Avg: ', str(np.round(annual_CV_RMSE[-1], 4)), '\nPWA PM25 for models: ', 'MAX: ',str(np.round(np.max(annual_CV_models), 4)),'Min: ',
                         str(np.round(np.min(annual_CV_models), 4)),
                         'Avg: ', str(np.round(annual_CV_models[-1], 4)),'\nPWA PM25 for monitors : ', 'MAX: ',str(np.round(np.max(annual_CV_monitors), 4)),'Min: ',
                         str(np.round(np.min(annual_CV_monitors), 4)),
                         'Avg: ', str(np.round(annual_CV_monitors[-1], 4))])
        
        for imonth in range(len(MONTH)):
            month_CV_R2[imonth,-1] = np.mean(month_CV_R2[imonth,0:kfold * repeats])
            month_CV_slope[imonth,-1] = np.mean(month_CV_slope[imonth,0:kfold * repeats])
            month_CV_RMSE[imonth,-1] = np.mean(month_CV_RMSE[imonth,0:kfold * repeats])
            month_CV_models[imonth,-1] = np.mean(month_CV_models[imonth,0:kfold * repeats])
            month_CV_monitors[imonth,-1] = np.mean(month_CV_monitors[imonth,0:kfold * repeats])
            writer.writerow(['-------------------------- {} ------------------------'.format(MONTH[imonth]),'\nR2 - Max: ', str(np.round(np.max(month_CV_R2[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_R2[imonth,-1],4)),'\n Slope Max: ', str(np.round(np.max(month_CV_slope[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_slope[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_slope[imonth,-1],4)),'\n RMSE Max: ', str(np.round(np.max(month_CV_RMSE[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_RMSE[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_RMSE[imonth,-1],4)),'\n PWA PM25 for models Max: ', str(np.round(np.max(month_CV_models[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_models[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_models[imonth,-1],4)),'\n PWA PM25 for monitors Max: ', str(np.round(np.max(month_CV_monitors[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_monitors[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_monitors[imonth,-1],4))])
    return

def Optimal_Model_Output_Text(outfile:str,status:str,training_annual_CV_R2,training_month_CV_R2,CV_R2,annual_CV_R2,month_CV_R2,CV_slope,annual_CV_slope,month_CV_slope,
                CV_RMSE,annual_CV_RMSE,month_CV_RMSE,annual_CV_models,annual_CV_monitors,month_CV_models,month_CV_monitors,beginyear:str,
                endyear:str,Area:str,kfold:int,repeats:int):
    MONTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    CV_R2[-1] = np.mean(CV_R2[0:kfold * repeats])
    annual_CV_R2[-1] = np.mean(annual_CV_R2[0:kfold * repeats])
    CV_slope[-1] = np.mean(CV_slope[0:kfold * repeats])
    training_annual_CV_R2[-1] = np.mean(training_annual_CV_R2[0:kfold * repeats])
    annual_CV_slope[-1] = np.mean(annual_CV_slope[0:kfold * repeats])
    CV_RMSE[-1] = np.mean(CV_RMSE[0:kfold * repeats])
    annual_CV_RMSE[-1] = np.mean(annual_CV_RMSE[0:kfold * repeats])

    annual_CV_models[-1] = np.mean(annual_CV_models[0:kfold * repeats])
    annual_CV_monitors[-1] = np.mean(annual_CV_monitors[0:kfold * repeats])
   

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Area,'Area ','Time Period: ', beginyear,' ', endyear])
        writer.writerow(['R2 for monthly validation','Max: ',str(np.round(np.max(CV_R2),4)),'Min: ',str(np.round(np.min(CV_R2),4)),
                         'Avg: ',str(np.round(CV_R2[-1],4)),'\nSlope for monthly validation','Max: ',str(np.round(np.max(CV_slope),4)),'Min: ',str(np.round(np.min(CV_slope),4)),
                         'Avg: ',str(np.round(CV_slope[-1],4)),'\nRMSE for monthly validation','Max: ',str(np.round(np.max(CV_RMSE),4)),'Min: ',str(np.round(np.min(CV_RMSE),4)),
                         'Avg: ',str(np.round(CV_RMSE[-1],4))])
        writer.writerow(['#####################   Annual average validation ####################','\nR2 - Max: ', str(np.round(np.max(annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_R2), 4)),
                         'Avg: ', str(np.round(annual_CV_R2[-1], 4)),'\n Slope for Annual average validation', 'Max: ', str(np.round(np.max(annual_CV_slope), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_slope), 4)),
                         'Avg: ', str(np.round(annual_CV_slope[-1], 4)),'\n RMSE for Annual average validation', 'Max: ', str(np.round(np.max(annual_CV_RMSE), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_RMSE), 4)),
                         'Avg: ', str(np.round(annual_CV_RMSE[-1], 4)), '\n PWA PM25 for models: ', 'MAX: ',str(np.round(np.max(annual_CV_models), 4)),'Min: ',
                         str(np.round(np.min(annual_CV_models), 4)),
                         'Avg: ', str(np.round(annual_CV_models[-1], 4)),'\n PWA PM25 for monitors : ', 'MAX: ',str(np.round(np.max(annual_CV_monitors), 4)),'Min: ',
                         str(np.round(np.min(annual_CV_monitors), 4)),
                         'Avg: ', str(np.round(annual_CV_monitors[-1], 4))])
        writer.writerow(['###################### Annual Training ####################','\nR2 - Max: ', str(np.round(np.max(training_annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(training_annual_CV_R2), 4)),
                         'Avg: ', str(np.round(training_annual_CV_R2[-1], 4))])
        
        for imonth in range(len(MONTH)):
            month_CV_R2[imonth,-1] = np.mean(month_CV_R2[imonth,0:kfold * repeats])
            month_CV_slope[imonth,-1] = np.mean(month_CV_slope[imonth,0:kfold * repeats])
            month_CV_RMSE[imonth,-1] = np.mean(month_CV_RMSE[imonth,0:kfold * repeats])
            month_CV_models[imonth,-1] = np.mean(month_CV_models[imonth,0:kfold * repeats])
            month_CV_monitors[imonth,-1] = np.mean(month_CV_monitors[imonth,0:kfold * repeats])
            training_month_CV_R2[imonth,-1] = np.mean(training_month_CV_R2[imonth,0:kfold * repeats])
            writer.writerow([' -------------------------- {} ------------------------'.format(MONTH[imonth]),'R2 - Max: ', str(np.round(np.max(month_CV_R2[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_R2[imonth,-1],4)),' Slope Max: ', str(np.round(np.max(month_CV_slope[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_slope[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_slope[imonth,-1],4)),' RMSE Max: ', str(np.round(np.max(month_CV_RMSE[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_RMSE[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_RMSE[imonth,-1],4)),'PWA PM25 for models Max: ', str(np.round(np.max(month_CV_models[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_models[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_models[imonth,-1],4)),'PWA PM25 for monitors Max: ', str(np.round(np.max(month_CV_monitors[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_monitors[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_monitors[imonth,-1],4)),'Training Max: ',str(np.round(np.max(training_month_CV_R2[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(training_month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(training_month_CV_R2[imonth,-1],4))])
    return

def Optimal_Model_GBDAreaAllfolds_Output_Text(outfile:str,status:str,training_annual_CV_R2,
                annual_CV_R2,beginyear:str,
                endyear:str,Area:str,kfold:int,repeats:int):
    MONTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    training_annual_CV_R2[-1] = np.mean(training_annual_CV_R2[0:kfold * repeats])
    
    

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Area,'Area ','Time Period: ', beginyear,' ', endyear])
        
        writer.writerow(['R2 for Annual average validation', 
                         'Avg: ', str(np.round(annual_CV_R2, 4))])
        writer.writerow(['R2 for Annual Training', 'Max: ', str(np.round(np.max(training_annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(training_annual_CV_R2), 4)),
                         'Avg: ', str(np.round(training_annual_CV_R2[-1], 4))])
        
        
    return
def CalculateMonthR2(test_index,final_data,test_obs_data,population,beginyear:int,endyear:int):
    '''
    This funciton is used to calculate the monthly R2, slope and RMSE
    return:
    month_R2, month_slope, month_RMSE
    '''
    month_obs = np.zeros((len(test_index)))
    month_predict = np.zeros((len(test_index)))
    month_population = np.zeros(len(test_index))

    monthly_test_month = np.array(range(endyear - beginyear + 1)) * 12
    month_R2 = np.zeros(12,dtype = np.float64)
    month_slope = np.zeros(12,dtype = np.float64)
    month_RMSE = np.zeros(12,dtype = np.float64)
    month_PWA_model = np.zeros(12,dtype = np.float64)
    month_PWA_monitor = np.zeros(12,dtype = np.float64)

    for imonth in range(12):
        for isite in range(len(test_index)):
            month_obs[isite] = np.mean(test_obs_data[isite + (imonth + monthly_test_month) * len(test_index)])
            month_predict[isite] = np.mean(final_data[isite + (imonth + monthly_test_month) * len(test_index)])
            month_population[isite] = np.mean(population[isite + (imonth + monthly_test_month) * len(test_index)])
        month_R2[imonth] = linear_regression(month_obs, month_predict)
        regression_Dic = regress2(_x=month_obs,_y=month_predict,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
        intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
        month_slope[imonth] = round(slope, 2)
        month_RMSE[imonth] = Cal_RMSE(month_obs, month_predict)
        month_PWA_model[imonth] = Calculate_PWA_PM25(month_population,month_predict)
        month_PWA_monitor[imonth] = Calculate_PWA_PM25(month_population,month_obs)
    return month_R2, month_slope, month_RMSE, month_PWA_model, month_PWA_monitor

def CalculateMonthR2_EachYear(test_index,final_data,test_obs_data,population,beginyear:int,endyear:int):
    '''
    This funciton is used to calculate the monthly R2, slope and RMSE
    return:
    month_R2, month_slope, month_RMSE
    '''
    month_obs = np.zeros((len(test_index)))
    month_predict = np.zeros((len(test_index)))
    month_population = np.zeros(len(test_index))
    month_R2 = np.zeros(12,dtype = np.float64)
    month_slope = np.zeros(12,dtype = np.float64)
    month_RMSE = np.zeros(12,dtype = np.float64)
    month_PWA_model = np.zeros(12,dtype = np.float64)
    month_PWA_monitor = np.zeros(12,dtype = np.float64)
    for imonth in range(12):
        temp_R2 = np.zeros((endyear-beginyear+1),dtype = np.float64)
        temp_slope = np.zeros((endyear-beginyear+1),dtype = np.float64)
        temp_RMSE = np.zeros((endyear-beginyear+1),dtype = np.float64)    
        temp_PWA_model = np.zeros((endyear-beginyear+1),dtype = np.float64)    
        temp_PWA_monitor = np.zeros((endyear-beginyear+1),dtype = np.float64)    
        for iyear in range(endyear-beginyear+1):
            monthly_test_month = iyear * 12
            
            for isite in range(len(test_index)):
                month_obs[isite] = test_obs_data[isite + (imonth + monthly_test_month) * len(test_index)]
                month_predict[isite] =final_data[isite + (imonth + monthly_test_month) * len(test_index)]
                month_population[isite] = population[isite + (imonth + monthly_test_month) * len(test_index)]
            
            temp_R2[iyear] = linear_regression(month_obs, month_predict)
            regression_Dic = regress2(_x=month_obs,_y=month_predict,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            intercept,temp_slope[iyear] = regression_Dic['intercept'], regression_Dic['slope']
            temp_RMSE[iyear] = Cal_RMSE(month_obs, month_predict)
            temp_PWA_model[iyear] =  Calculate_PWA_PM25(month_population,month_predict)
            temp_PWA_monitor[iyear] = Calculate_PWA_PM25(month_population,month_obs)

        month_R2[imonth] = np.mean(temp_R2)
        print( "r-squared: ", month_R2[imonth])
        slope = np.mean(temp_slope)
        month_slope[imonth] = round(slope, 2)
        month_RMSE[imonth] = np.mean(temp_RMSE)
        month_PWA_model[imonth] = np.mean(temp_PWA_model)
        month_PWA_monitor[imonth] = np.mean(temp_PWA_monitor)
    return month_R2, month_slope, month_RMSE, month_PWA_model, month_PWA_monitor


def CalculateTrainingMonthR2(train_index,final_training_data,train_obs_data,beginyear:int,endyear:int):
    '''
    This funciton is used to calculate the monthly R2, slope and RMSE
    return:
    month_R2, month_slope, month_RMSE
    '''
    month_obs = np.zeros((len(train_index)))
    month_predict = np.zeros((len(train_index)))
    monthly_test_month = np.array(range(endyear - beginyear + 1)) * 12
    month_R2 = np.zeros(12,dtype = np.float64)
    month_slope = np.zeros(12,dtype = np.float64)
    month_RMSE = np.zeros(12,dtype = np.float64)
    for imonth in range(12):
        for isite in range(len(train_index)):
            month_obs[isite] = np.mean(train_obs_data[isite + (imonth + monthly_test_month) * len(train_index)])
            month_predict[isite] = np.mean(final_training_data[isite + (imonth + monthly_test_month) * len(train_index)])
        month_R2[imonth] = linear_regression(month_obs, month_predict)
        regression_Dic = regress2(_x=month_obs,_y=month_predict,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
        intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
        #month_slope[imonth] = round(slope, 2)
        #month_RMSE = Cal_RMSE(month_obs, month_predict)
    return month_R2

def CalculateAnnualR2(test_index,final_data,population,test_obs_data,beginyear,endyear):
    '''
    This funciton is used to calculate the Annual R2, slope and RMSE
    return:
    annual_R2,annual_final_data,annual_mean_obs,slope, RMSE
    '''
    annual_mean_obs = np.zeros((len(test_index)))
    annual_final_data = np.zeros((len(test_index)))
    annual_population = np.zeros((len(test_index)))
    test_month = np.array(range((endyear - beginyear + 1) * 12))
    for isite in range(len(test_index)):
        annual_mean_obs[isite] = np.mean(test_obs_data[isite + test_month * len(test_index)])
        annual_final_data[isite] = np.mean(final_data[isite + test_month * len(test_index)])
        annual_population[isite] = np.mean(population[isite + test_month * len(test_index)])
    annual_R2 = linear_regression(annual_mean_obs, annual_final_data)
    regression_Dic = regress2(_x=annual_mean_obs,_y=annual_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',
    )
    intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    intercept = round(intercept, 2)
    slope = round(slope, 2)
    RMSE = Cal_RMSE(annual_mean_obs, annual_final_data)
    annual_PWA_model = Calculate_PWA_PM25(annual_population,annual_final_data)
    annual_PWA_monitor = Calculate_PWA_PM25(annual_population,annual_mean_obs)
    return annual_R2,annual_final_data,annual_mean_obs,slope,RMSE,annual_PWA_model,annual_PWA_monitor


def CalculateAnnualR2_EachYear(test_index,final_data,population,test_obs_data,beginyear,endyear):
    annual_mean_obs = np.zeros((len(test_index)))
    annual_final_data = np.zeros((len(test_index)))
    annual_population = np.zeros((len(test_index)))
    temp_R2 = np.zeros((endyear-beginyear+1),dtype = np.float64)    
    temp_slope = np.zeros((endyear-beginyear+1),dtype = np.float64)    
    temp_RMSE = np.zeros((endyear-beginyear+1),dtype = np.float64)    
    temp_PWA_model = np.zeros((endyear-beginyear+1),dtype = np.float64)    
    temp_PWA_monitor = np.zeros((endyear-beginyear+1),dtype = np.float64)    
    for iyear in range((endyear - beginyear + 1)):
        test_month = np.array(range(12)) + iyear*12
        for isite in range(len(test_index)):
            annual_mean_obs[isite]   = np.mean(test_obs_data[isite + test_month * len(test_index)])
            annual_final_data[isite] = np.mean(final_data[isite + test_month * len(test_index)])
            annual_population[isite] = np.mean(population[isite + test_month * len(test_index)])
        temp_R2[iyear] = linear_regression(annual_mean_obs, annual_final_data)
        regression_Dic = regress2(_x=annual_mean_obs,_y=annual_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
        intercept,temp_slope[iyear] = regression_Dic['intercept'], regression_Dic['slope']
        temp_RMSE[iyear] = Cal_RMSE(annual_mean_obs, annual_final_data)
        temp_PWA_model[iyear] =  Calculate_PWA_PM25(annual_population,annual_final_data)
        temp_PWA_monitor[iyear] = Calculate_PWA_PM25(annual_population,annual_mean_obs)
    annual_R2 = np.mean(temp_R2)
   
    # b0, b1 = linear_slope(plot_obs_pm25, plot_pre_pm25)
    slope = np.mean(temp_slope)
    slope = round(slope, 2)
    RMSE = np.mean(temp_RMSE)
    annual_PWA_model = np.mean(temp_PWA_model)
    annual_PWA_monitor = np.mean(temp_PWA_monitor)
    return annual_R2,annual_final_data,annual_mean_obs,slope,RMSE,annual_PWA_model,annual_PWA_monitor


def derive_Annual_data(test_index,final_data,population,test_obs_data,beginyear,endyear):
    annual_mean_obs = np.zeros((len(test_index)))
    annual_final_data = np.zeros((len(test_index)))
    test_month = np.array(range((endyear - beginyear + 1) * 12))
    for isite in range(len(test_index)):
        annual_mean_obs[isite] = np.mean(test_obs_data[isite + test_month * len(test_index)])
        annual_final_data[isite] = np.mean(final_data[isite + test_month * len(test_index)])
    
    return annual_final_data,annual_mean_obs


def CalculateTrainingAnnualR2(train_index,train_final_data,train_obs_data,beginyear,endyear):
    '''
    This funciton is used to calculate the Annual R2, slope and RMSE
    return:
    annual_R2,annual_final_data,annual_mean_obs,slope, RMSE
    '''
    print('Length of Train index: ', len(train_index), '\n length of train_final_data: ',len(train_final_data),
          '\n beginyear: ',beginyear, ' endyear: ',endyear)
    annual_mean_obs = np.zeros((len(train_index)))
    annual_final_data = np.zeros((len(train_index)))
    test_month = np.array(range((endyear - beginyear + 1) * 12))
    for isite in range(len(train_index)):
        annual_mean_obs[isite] = np.mean(train_obs_data[isite + test_month * len(train_index)])
        annual_final_data[isite] = np.mean(train_final_data[isite + test_month * len(train_index)])
    annual_R2 = linear_regression(annual_mean_obs, annual_final_data)
    return annual_R2

def ForcedSlopeUnity_Func(train_final_data,train_obs_data,test_final_data,train_area_index,test_area_index,endyear,beginyear,EachMonth:bool):
    if EachMonth:
        for i in range(12 * (endyear - beginyear + 1)):
            temp_train_final_data = train_final_data[i*len(train_area_index):(i+1)*len(train_area_index)]
            temp_train_obs_data   = train_obs_data[i*len(train_area_index):(i+1)*len(train_area_index)]
            temp_regression_dic = regress2(_x=temp_train_obs_data,_y=temp_train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            temp_offset,temp_slope = temp_regression_dic['intercept'], temp_regression_dic['slope']
            test_final_data[i*len(test_area_index):(i+1)*len(test_area_index)] = (test_final_data[i*len(test_area_index):(i+1)*len(test_area_index)] - temp_offset)/temp_slope
    else:
        month_train_obs_average = np.zeros((len(train_area_index)))
        month_train_average = np.zeros((len(train_area_index)))
        monthly_test_month = np.array(range(endyear - beginyear + 1)) * 12
        for imonth in range(12):
            for isite in range(len(train_area_index)):
                month_train_obs_average[isite] = np.mean(train_final_data[isite + (imonth + monthly_test_month) * len(train_area_index)])
                month_train_average[isite] = np.mean(train_final_data[isite + (imonth + monthly_test_month) * len(train_area_index)])
            temp_regression_dic = regress2(_x=month_train_obs_average,_y=month_train_average,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            temp_offset,temp_slope = temp_regression_dic['intercept'], temp_regression_dic['slope']

            for iyear in range(endyear-beginyear+1):
                test_final_data[(iyear*12+imonth)*len(test_area_index):(iyear*12+imonth+1)*len(test_area_index)] -= temp_offset
                test_final_data[(iyear*12+imonth)*len(test_area_index):(iyear*12+imonth+1)*len(test_area_index)] /= temp_slope
    return test_final_data
            


def GetXYIndex(Global_index,area_index,train_index,beginyear:int, endyear:int,
                      databeginyear:int,GLsitesNum:int):
    '''
    This funcion is to get the X_index, Y_index for Cross-Validation in each time period.
    :param area_index: The index of sites of this specific area.
    :param train_index: train_index of this fold.
    :param test_index: test_index of this fold.
    :param beginyear: The beginyear of this model.
    :param endyear: The endyear of this model.
    :param databeginyear: The beginyear of the datasets.
    :param GLsitesNum: Number of global sites
    :return:
    '''
    X_index = np.zeros((12 * (endyear - beginyear + 1) * len(train_index)), dtype=int)
    Y_index = np.zeros((12 * (endyear - beginyear + 1) * len(area_index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        X_index[i * len(train_index):(i + 1) * len(train_index)] = ((beginyear - databeginyear) * 12 + i) * GLsitesNum + \
                                                                   Global_index[train_index]
        Y_index[i * len(area_index):(i + 1) * len(area_index)] = ((beginyear - 1998) * 12 + i) * GLsitesNum + area_index
    return X_index,Y_index

def GetTrainingIndex(Global_index,train_index,beginyear:int, endyear:int,
                      databeginyear:int,GLsitesNum:int):
    X_index = np.zeros((12 * (endyear - beginyear + 1) * len(train_index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        X_index[i * len(train_index):(i + 1) * len(train_index)] = ((beginyear - databeginyear) * 12 + i) * GLsitesNum + \
                                                                   Global_index[train_index]
    return X_index       

def GetValidationIndex(area_index,beginyear:int, endyear:int,GLsitesNum:int):
    Y_index = np.zeros((12 * (endyear - beginyear + 1) * len(area_index)), dtype=int)  
    for i in range(12 * (endyear - beginyear + 1)):  
        Y_index[i * len(area_index):(i + 1) * len(area_index)] = ((beginyear - 1998) * 12 + i) * GLsitesNum + area_index
    return Y_index                                

def Initialize_DataRecording_Dic(breakpoints):
    annual_final_test = {}
    annual_obs_test   = {}
    for imodel in range(len(breakpoints)):
        annual_final_test[str(breakpoints[imodel])] = np.array([],dtype=np.float64)
        annual_obs_test[str(breakpoints[imodel])] = np.array([],dtype=np.float64)

    annual_final_test['Alltime'] =  np.array([],dtype=np.float64)
    annual_obs_test['Alltime']   =  np.array([],dtype=np.float64)
    return annual_final_test,annual_obs_test

def Initialize_DataRecording_MultiAreas_Dic(breakpoints, MultiyearsForAreas:list):
    annual_final_test = {}
    annual_obs_test   = {}
    for imodel in range(len(breakpoints)):
        annual_final_test[str(breakpoints[imodel])] ={}
        annual_obs_test[str(breakpoints[imodel])] = {}
        annual_final_test['Alltime'] = {}
        annual_obs_test['Alltime'] = {}
        for iarea in range(len(MultiyearsForAreas[imodel])):
            annual_final_test[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.array([],dtype=np.float64)
            annual_obs_test[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.array([],dtype=np.float64)
            annual_final_test['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.array([],dtype=np.float64)
            annual_obs_test['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.array([],dtype=np.float64)
    return annual_final_test,annual_obs_test

def Initialize_CV_Dic(kfold:int, repeats:int, breakpoints):
    '''

    The function is to initialize the CV recording Dictionaries
    :param kfold: k number of folds
    :param repeats: repeat time
    :param breakpoints:
    :return: CV_R2 - record the R2 for each fold(original data).
             annual_CV_R2 - record the R2 for purely spatial R2.
             month_CV_R2  - record the R2 for purely spatial R2 for months.
    '''
    CV_R2 = {}
    CV_slope = {}
    CV_RMSE = {}
    annual_CV_R2 = {}
    month_CV_R2 = {}
    annual_CV_slope = {}
    month_CV_slope = {}
    annual_CV_RMSE = {}
    month_CV_RMSE = {}

    for imodel in range(len(breakpoints)):
        CV_R2[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_R2[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        month_CV_R2[str(breakpoints[imodel])] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
        CV_slope[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_slope[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        month_CV_slope[str(breakpoints[imodel])] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
        CV_RMSE[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_RMSE[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        month_CV_RMSE[str(breakpoints[imodel])] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    
    CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    month_CV_R2['Alltime'] = np.zeros((12, kfold * repeats + 1), dtype=np.float32)
    CV_slope['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_slope['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    month_CV_slope['Alltime'] = np.zeros((12, kfold * repeats + 1), dtype=np.float32)
    CV_RMSE['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_RMSE['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    month_CV_RMSE['Alltime'] = np.zeros((12, kfold * repeats + 1), dtype=np.float32)
    return CV_R2, annual_CV_R2, month_CV_R2, CV_slope,annual_CV_slope,month_CV_slope,CV_RMSE,annual_CV_RMSE,month_CV_RMSE
def Initialize_multiareas_CV_Dic(kfold:int, repeats:int, breakpoints, MultiyearsForAreas:list):
    '''
    The function is to initialize the CV recording Dictionaries in different years and different areas.
    :param kfold: k number of folds
    :param repeats: repeat time
    :param breakpoints:
    :return: CV_R2 - record the R2 for each fold(original data).
             annual_CV_R2 - record the R2 for purely spatial R2.
             month_CV_R2  - record the R2 for purely spatial R2 for months.
             CV_slope - record the slope for each fold(original data).
             annual_CV_slope - record the slope for purely spatial R2.
             month_CV_slope  - record the slope for purely spatial R2 for months.
             CV_RMSE - record the R2 for each fold(original data).
             annual_CV_RMSE - record the R2 for purely spatial R2.
             month_CV_RMSE  - record the R2 for purely spatial R2 for months.
    '''
    CV_R2 = {}
    CV_slope = {}
    CV_RMSE = {}
    annual_CV_R2 = {}
    month_CV_R2 = {}
    annual_CV_slope = {}
    month_CV_slope = {}
    annual_CV_RMSE = {}
    month_CV_RMSE = {}

    annual_CV_PWAModel = {}
    month_CV_PWAModel = {}
    annual_CV_PWAMonitor = {}
    month_CV_PWAMonitor = {}
    for imodel in range(len(breakpoints)):
        CV_R2[str(breakpoints[imodel])] = {}
        annual_CV_R2[str(breakpoints[imodel])] = {}
        month_CV_R2[str(breakpoints[imodel])] = {}
        CV_R2['Alltime'] = {}
        annual_CV_R2['Alltime'] = {}
        month_CV_R2['Alltime'] = {}

        CV_slope[str(breakpoints[imodel])] = {}
        annual_CV_slope[str(breakpoints[imodel])] = {}
        month_CV_slope[str(breakpoints[imodel])] = {}
        CV_slope['Alltime'] = {}
        annual_CV_slope['Alltime'] = {}
        month_CV_slope['Alltime'] = {}

        CV_RMSE[str(breakpoints[imodel])] = {}
        annual_CV_RMSE[str(breakpoints[imodel])] = {}
        month_CV_RMSE[str(breakpoints[imodel])] = {}
        CV_RMSE['Alltime'] = {}
        annual_CV_RMSE['Alltime'] = {}
        month_CV_RMSE['Alltime'] = {}

        annual_CV_PWAModel[str(breakpoints[imodel])] = {}
        annual_CV_PWAModel['Alltime'] = {}
        month_CV_PWAModel[str(breakpoints[imodel])] = {}
        month_CV_PWAModel['Alltime'] = {}
        annual_CV_PWAMonitor[str(breakpoints[imodel])] = {}
        annual_CV_PWAMonitor['Alltime'] = {}
        month_CV_PWAMonitor[str(breakpoints[imodel])] = {}
        month_CV_PWAMonitor['Alltime'] = {}


        for iarea in range(len(MultiyearsForAreas[imodel])):
            CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

            CV_slope[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_slope[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_slope[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            CV_slope['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_slope['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_slope['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

            CV_RMSE[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_RMSE[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_RMSE[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            CV_RMSE['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_RMSE['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_RMSE['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            
            annual_CV_PWAModel[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_PWAModel['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_PWAMonitor[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_PWAMonitor['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)

            month_CV_PWAModel[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12,kfold * repeats + 1), dtype=np.float32)
            month_CV_PWAModel['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            month_CV_PWAMonitor[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12,kfold * repeats + 1), dtype=np.float32)
            month_CV_PWAMonitor['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

    return CV_R2, annual_CV_R2, month_CV_R2, CV_slope, annual_CV_slope, month_CV_slope, CV_RMSE, annual_CV_RMSE, month_CV_RMSE, annual_CV_PWAModel,month_CV_PWAModel,annual_CV_PWAMonitor,month_CV_PWAMonitor


def Initialize_multiareas_optimalModel_CV_Dic(kfold:int, repeats:int, breakpoints, MultiyearsForAreas:list):
    '''
    The function is to initialize the CV recording Dictionaries in different years and different areas.
    :param kfold: k number of folds
    :param repeats: repeat time
    :param breakpoints:
    :return: CV_R2 - record the R2 for each fold(original data).
             annual_CV_R2 - record the R2 for purely spatial R2.
             month_CV_R2  - record the R2 for purely spatial R2 for months.
             CV_slope - record the slope for each fold(original data).
             annual_CV_slope - record the slope for purely spatial R2.
             month_CV_slope  - record the slope for purely spatial R2 for months.
             CV_RMSE - record the R2 for each fold(original data).
             annual_CV_RMSE - record the R2 for purely spatial R2.
             month_CV_RMSE  - record the R2 for purely spatial R2 for months.
    '''
    training_CV_R2 = {}
    training_annual_CV_R2 = {}
    training_month_CV_R2 = {}
    CV_R2 = {}
    CV_slope = {}
    CV_RMSE = {}
    annual_CV_R2 = {}
    month_CV_R2 = {}
    annual_CV_slope = {}
    month_CV_slope = {}
    annual_CV_RMSE = {}
    month_CV_RMSE = {}

    annual_CV_PWAModel = {}
    month_CV_PWAModel = {}
    annual_CV_PWAMonitor = {}
    month_CV_PWAMonitor = {}

    for imodel in range(len(breakpoints)):
        training_CV_R2[str(breakpoints[imodel])] = {}
        training_annual_CV_R2[str(breakpoints[imodel])] = {}
        training_month_CV_R2[str(breakpoints[imodel])] = {}
        training_CV_R2['Alltime'] = {}
        training_annual_CV_R2['Alltime'] = {}
        training_month_CV_R2['Alltime'] = {}

        CV_R2[str(breakpoints[imodel])] = {}
        annual_CV_R2[str(breakpoints[imodel])] = {}
        month_CV_R2[str(breakpoints[imodel])] = {}
        CV_R2['Alltime'] = {}
        annual_CV_R2['Alltime'] = {}
        month_CV_R2['Alltime'] = {}

        CV_slope[str(breakpoints[imodel])] = {}
        annual_CV_slope[str(breakpoints[imodel])] = {}
        month_CV_slope[str(breakpoints[imodel])] = {}
        CV_slope['Alltime'] = {}
        annual_CV_slope['Alltime'] = {}
        month_CV_slope['Alltime'] = {}

        CV_RMSE[str(breakpoints[imodel])] = {}
        annual_CV_RMSE[str(breakpoints[imodel])] = {}
        month_CV_RMSE[str(breakpoints[imodel])] = {}
        CV_RMSE['Alltime'] = {}
        annual_CV_RMSE['Alltime'] = {}
        month_CV_RMSE['Alltime'] = {}

        annual_CV_PWAModel[str(breakpoints[imodel])] = {}
        annual_CV_PWAModel['Alltime'] = {}
        month_CV_PWAModel[str(breakpoints[imodel])] = {}
        month_CV_PWAModel['Alltime'] = {}
        annual_CV_PWAMonitor[str(breakpoints[imodel])] = {}
        annual_CV_PWAMonitor['Alltime'] = {}
        month_CV_PWAMonitor[str(breakpoints[imodel])] = {}
        month_CV_PWAMonitor['Alltime'] = {}

        for iarea in range(len(MultiyearsForAreas[imodel])):
            training_CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            training_annual_CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            training_month_CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            training_CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            training_annual_CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            training_month_CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

            CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_R2[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_R2['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

            CV_slope[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_slope[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_slope[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            CV_slope['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_slope['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_slope['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

            CV_RMSE[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_RMSE[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_RMSE[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            CV_RMSE['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_RMSE['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            month_CV_RMSE['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    
            annual_CV_PWAModel[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_PWAModel['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_PWAMonitor[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)
            annual_CV_PWAMonitor['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((kfold * repeats + 1), dtype=np.float32)

            month_CV_PWAModel[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12,kfold * repeats + 1), dtype=np.float32)
            month_CV_PWAModel['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
            month_CV_PWAMonitor[str(breakpoints[imodel])][MultiyearsForAreas[imodel][iarea]] = np.zeros((12,kfold * repeats + 1), dtype=np.float32)
            month_CV_PWAMonitor['Alltime'][MultiyearsForAreas[imodel][iarea]] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

    return training_CV_R2, training_annual_CV_R2,training_month_CV_R2, CV_R2, annual_CV_R2, month_CV_R2, CV_slope, annual_CV_slope, month_CV_slope, CV_RMSE, annual_CV_RMSE, month_CV_RMSE,annual_CV_PWAModel,month_CV_PWAModel,annual_CV_PWAMonitor,month_CV_PWAMonitor


def Get_data_NormPara(input_dir:str,input_file:str):
    infile = input_dir + input_file
    data   = np.load(infile)
    data_mean = np.mean(data)
    data_std  = np.std(data)
    return data, data_mean, data_std

def Get_CV_seed():
    seed = 32456548
    print('Seed is :', seed)
    return seed


def initialize_weights_Xavier(m): #xavier 
  tanh_gain = nn.init.calculate_gain('tanh')
  if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight.data,gain=tanh_gain)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

