import numpy as np
from Training_pkg.Statistic_Func import linear_regression,regress2, Cal_RMSE, Cal_NRMSE,Cal_PWM_rRMSE,Calculate_PWA_PM25
from Evaluation_pkg.utils import *


#######################################################################################################################
# This part is for year-based training models. (For example, Datasets of 1998 Jan-Dec are used for training a model)

def find_masked_latlon(mask_map,mask_lat,mask_lon,test_lat,test_lon):
    index_lon,index_lat = get_nearest_point_index(test_lon,test_lat,mask_lon,mask_lat)
    masked_obs_array = mask_map[index_lat,index_lon]
    masked_array_index = np.where(masked_obs_array == 1)
    return masked_array_index[0]

def GetXIndex(index,beginyear:int, endyear:int, sitenumber:int):
    X_index = np.zeros((12 * (endyear - beginyear + 1) * len(index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        X_index[i * len(index):(i + 1) * len(index)] = i * sitenumber + index
    return X_index

def GetYIndex(index,beginyear:int, endyear:int,sitenumber:int):
    # Y is for observations
    Y_index = np.zeros((12 * (endyear - beginyear + 1) * len(index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        Y_index[i * len(index):(i + 1) * len(index)] = ((beginyear - 1998)*12 + i) * sitenumber + index
    return Y_index   

def Get_XY_indices(train_index,test_index, beginyear, endyear, sitesnumber ):
    X_Training_index = GetXIndex(index=train_index,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    X_Testing_index  = GetXIndex(index=test_index ,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    Y_Training_index = GetYIndex(index=train_index,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    Y_Testing_index  = GetYIndex(index=test_index ,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    return X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index

# This part is for month-based training models. (For example, Datasets of 1998-2003 Jan-Mar are used for training a model.)

def Get_month_based_XIndex(index,beginyear:int, endyear:int, month_index:np.array, sitenumber:int):
    X_index = np.zeros((len(month_index) * (endyear - beginyear + 1) * len(index)), dtype=int)
    for iyear in range(endyear - beginyear + 1):
        for imonth in range(len(month_index)):
            X_index[(iyear*len(month_index)+imonth) * len(index):(iyear*len(month_index)+imonth + 1) * len(index)] = (iyear*12+month_index[imonth]) * sitenumber + index
    return X_index

def Get_month_based_YIndex(index,beginyear:int, endyear:int,month_index:np.array, sitenumber:int):
    # Y is for observations
    Y_index = np.zeros((len(month_index) * (endyear - beginyear + 1) * len(index)), dtype=int)
    for iyear in range(endyear - beginyear + 1):
        for imonth in range(len(month_index)):
            Y_index[(iyear*len(month_index)+imonth) * len(index):((iyear*len(month_index)+imonth) + 1) * len(index)] = ((beginyear - 1998 + iyear)*12 + month_index[imonth]) * sitenumber + index
    return Y_index

def Get_month_based_XY_indices(train_index,test_index, beginyear, endyear, month_index, sitesnumber ):
    X_Training_index = Get_month_based_XIndex(index=train_index,beginyear=beginyear,endyear=endyear,month_index=month_index,sitenumber=sitesnumber)
    X_Testing_index  = Get_month_based_XIndex(index=test_index ,beginyear=beginyear,endyear=endyear,month_index=month_index,sitenumber=sitesnumber)
    Y_Training_index = Get_month_based_YIndex(index=train_index,beginyear=beginyear,endyear=endyear,month_index=month_index,sitenumber=sitesnumber)
    Y_Testing_index  = Get_month_based_YIndex(index=test_index ,beginyear=beginyear,endyear=endyear,month_index=month_index,sitenumber=sitesnumber)
    return X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index

def Get_XY_arraies(Normalized_TrainingData, true_input, X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index):
    print('length of Normalized_TrainingData: {}, length of true_input : {}, \nlength of X_Training_index: {}, length of Y_Training_index: {},\
          \n length of X_Testing_index: {}, length of Y_Testing_index: {}'.format(len(Normalized_TrainingData), len(true_input),\
                                                                                   len(X_Training_index), len(Y_Training_index),\
                                                                                    len(X_Testing_index), len(Y_Testing_index)))
    X_train, y_train  = Normalized_TrainingData[X_Training_index, :, :, :], true_input[Y_Training_index]
    X_test,  y_test   = Normalized_TrainingData[X_Testing_index, :, :, :], true_input[Y_Testing_index]
    return X_train, X_test, y_train, y_test
    

def Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,Y_Testing_index  ):
    """This function is used to convert the model estimation to absolute PM species concentration and to compare with the 
    observed PM species.

    Args:
        Validation_Prediction (_type_): _description_
        geophysical_species (_type_): _description_
        SPECIES_OBS (_type_): _description_
        bias (_type_): _description_
        normalize_species (_type_): _description_
        absolute_species (_type_): _description_
        log_species (_type_): _description_
        Y_Testing_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    if bias == True:
        final_data = Validation_Prediction + geophysical_species[Y_Testing_index]
    elif normalize_bias == True:
        final_data = Validation_Prediction * std + mean + geophysical_species[Y_Testing_index]
    elif normalize_species == True:
        final_data = Validation_Prediction * std + mean
    elif absolute_species == True:
        final_data = Validation_Prediction
    elif log_species == True:
        final_data = np.exp(Validation_Prediction) - 1
    return final_data



def ForcedSlopeUnity_Func(train_final_data,train_obs_data,test_final_data,train_area_index,test_area_index,endyear,beginyear,month_index, EachMonth):
    if EachMonth:
        for i in range(len(month_index) * (endyear - beginyear + 1)):
            temp_train_final_data = train_final_data[i*len(train_area_index):(i+1)*len(train_area_index)]
            temp_train_obs_data   = train_obs_data[i*len(train_area_index):(i+1)*len(train_area_index)]
            temp_regression_dic = regress2(_x=temp_train_obs_data,_y=temp_train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            temp_offset,temp_slope = temp_regression_dic['intercept'], temp_regression_dic['slope']
            test_final_data[i*len(test_area_index):(i+1)*len(test_area_index)] = (test_final_data[i*len(test_area_index):(i+1)*len(test_area_index)] - temp_offset)/temp_slope
    else:
        month_train_obs_average = np.zeros((len(train_area_index)))
        month_train_average = np.zeros((len(train_area_index)))
        monthly_test_month = np.array(range(endyear - beginyear + 1)) * len(month_index)
        for imonth in range(len(month_index)):
            for isite in range(len(train_area_index)):
                month_train_obs_average[isite] = np.mean(train_final_data[isite + (imonth + monthly_test_month) * len(train_area_index)])
                month_train_average[isite] = np.mean(train_final_data[isite + (imonth + monthly_test_month) * len(train_area_index)])
            temp_regression_dic = regress2(_x=month_train_obs_average,_y=month_train_average,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            temp_offset,temp_slope = temp_regression_dic['intercept'], temp_regression_dic['slope']

            for iyear in range(endyear-beginyear+1):
                test_final_data[(iyear*len(month_index)+imonth)*len(test_area_index):(iyear*len(month_index)+imonth+1)*len(test_area_index)] -= temp_offset
                test_final_data[(iyear*len(month_index)+imonth)*len(test_area_index):(iyear*len(month_index)+imonth+1)*len(test_area_index)] /= temp_slope
    return test_final_data



def calculate_Statistics_results(test_beginyear,test_endyear:int,final_data_recording, obs_data_recording, geo_data_recording, training_final_data_recording, training_obs_data_recording,testing_population_data_recording,masked_array_index):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Season_MONTH = [['Mar', 'Apr', 'May'], ['Jun', 'Jul', 'Aug'], ['Sep', 'Oct', 'Nov'], ['Dec','Jan', 'Feb']]
    Seasons = ['MAM','JJA','SON','DJF']

    All_points_obs_data_recording   = {}
    All_points_geo_data_recording   = {}
    All_points_final_data_recording = {}
    All_points_train_final_data_recording = {}
    All_points_train_obs_data_recording = {}
    All_points_population_data_recording = {}

    All_points_obs_data_recording['AllPoints']   = np.zeros([],dtype=np.float32)
    All_points_geo_data_recording['AllPoints']   = np.zeros([],dtype=np.float32)
    All_points_final_data_recording['AllPoints'] = np.zeros([],dtype=np.float32)
    All_points_train_final_data_recording['AllPoints'] = np.zeros([],dtype=np.float32)
    All_points_train_obs_data_recording['AllPoints'] = np.zeros([],dtype=np.float32)
    All_points_population_data_recording['AllPoints'] = np.zeros([],dtype=np.float32)

    All_points_obs_data_recording['Annual']   = np.zeros([],dtype=np.float32)
    All_points_geo_data_recording['Annual']   = np.zeros([],dtype=np.float32)
    All_points_final_data_recording['Annual'] = np.zeros([],dtype=np.float32)
    All_points_train_final_data_recording['Annual'] = np.zeros([],dtype=np.float32)
    All_points_train_obs_data_recording['Annual'] = np.zeros([],dtype=np.float32)
    All_points_population_data_recording['Annual'] = np.zeros([],dtype=np.float32)
        
    for imonth in MONTH:
        All_points_obs_data_recording[imonth]   = np.zeros([],dtype=np.float32)
        All_points_geo_data_recording[imonth]   = np.zeros([],dtype=np.float32)
        All_points_final_data_recording[imonth] = np.zeros([],dtype=np.float32)
        All_points_train_final_data_recording[imonth] = np.zeros([],dtype=np.float32)
        All_points_train_obs_data_recording[imonth] = np.zeros([],dtype=np.float32)
        All_points_population_data_recording[imonth] = np.zeros([],dtype=np.float32)

    for iseason in Seasons:
        All_points_obs_data_recording[iseason]   = np.zeros([],dtype=np.float32)
        All_points_geo_data_recording[iseason]   = np.zeros([],dtype=np.float32)
        All_points_final_data_recording[iseason] = np.zeros([],dtype=np.float32)
        All_points_train_final_data_recording[iseason] = np.zeros([],dtype=np.float32)
        All_points_train_obs_data_recording[iseason] = np.zeros([],dtype=np.float32)
        All_points_population_data_recording[iseason] = np.zeros([],dtype=np.float32)
    
                    
    test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE,slopes, PWAModel, PWAMonitors = initialize_AVD_CV_dict(test_beginyear=test_beginyear,test_endyear=test_endyear)
    if len(masked_array_index)!=0:
        for iyear in range(test_endyear-test_beginyear+1):
                for imonth in MONTH:
                    ### All Points and Monthly All Points Recording
                    All_points_obs_data_recording['AllPoints']   = np.append(All_points_obs_data_recording['AllPoints'],obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_geo_data_recording['AllPoints']   = np.append(All_points_geo_data_recording['AllPoints'],geo_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_final_data_recording['AllPoints'] = np.append(All_points_final_data_recording['AllPoints'],final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_train_final_data_recording['AllPoints'] = np.append(All_points_train_final_data_recording['AllPoints'],training_final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_train_obs_data_recording['AllPoints'] = np.append(All_points_train_obs_data_recording['AllPoints'],training_obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_population_data_recording['AllPoints'] = np.append(All_points_population_data_recording['AllPoints'],testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    
                    All_points_obs_data_recording[imonth]   = np.append(All_points_obs_data_recording[imonth],obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_geo_data_recording[imonth]   = np.append(All_points_geo_data_recording[imonth],geo_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_final_data_recording[imonth] = np.append(All_points_final_data_recording[imonth],final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_train_final_data_recording[imonth] = np.append(All_points_train_final_data_recording[imonth],training_final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_train_obs_data_recording[imonth] = np.append(All_points_train_obs_data_recording[imonth],training_obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    All_points_population_data_recording[imonth] = np.append(All_points_population_data_recording[imonth],testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])

                    ### Monthly Statistics Calculation for AVD Spatial CV
                    print('Area: {}, Year: {}, Month: {}'.format('NA', test_beginyear+iyear, imonth))
                    test_CV_R2[str(test_beginyear+iyear)][imonth] = linear_regression(final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    train_CV_R2[str(test_beginyear+iyear)][imonth] = linear_regression(training_final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], training_obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    geo_CV_R2[str(test_beginyear+iyear)][imonth] = linear_regression(geo_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    RMSE[str(test_beginyear+iyear)][imonth] = Cal_RMSE(final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    PWM_NRMSE[str(test_beginyear+iyear)][imonth] = Cal_PWM_rRMSE(final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index],testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    NRMSE[str(test_beginyear+iyear)][imonth] = Cal_NRMSE(final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    
                    regression_Dic = regress2(_x= obs_data_recording[str(test_beginyear+iyear)][imonth],_y=final_data_recording[str(test_beginyear+iyear)][imonth],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                    intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
                    slopes[str(test_beginyear+iyear)][imonth] = slope
                    PWAModel[str(test_beginyear+iyear)][imonth] = Calculate_PWA_PM25(Population_array=testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index],PM25_array=final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    PWAMonitors[str(test_beginyear+iyear)][imonth] = Calculate_PWA_PM25(Population_array=testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index],PM25_array=obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])

                    #### Get Annual Average Data Recording
                    if imonth == 'Jan':
                        final_data_recording[str(test_beginyear+iyear)]['Annual'] = final_data_recording[str(test_beginyear+iyear)][imonth].copy()
                        obs_data_recording[str(test_beginyear+iyear)]['Annual'] = obs_data_recording[str(test_beginyear+iyear)][imonth].copy()
                        geo_data_recording[str(test_beginyear+iyear)]['Annual'] = geo_data_recording[str(test_beginyear+iyear)][imonth].copy()
                        training_final_data_recording[str(test_beginyear+iyear)]['Annual'] = training_final_data_recording[str(test_beginyear+iyear)][imonth].copy()
                        testing_population_data_recording[str(test_beginyear+iyear)]['Annual'] = testing_population_data_recording[str(test_beginyear+iyear)][imonth].copy()
                        training_obs_data_recording[str(test_beginyear+iyear)]['Annual'] = training_obs_data_recording[str(test_beginyear+iyear)][imonth].copy()
                        #PWAModel[str(test_beginyear+iyear)]['Annual']  = PWAModel[str(test_beginyear+iyear)][imonth]
                        #PWAMonitors[str(test_beginyear+iyear)]['Annual'] = PWAMonitors[str(test_beginyear+iyear)][imonth]
                    else:
                        final_data_recording[str(test_beginyear+iyear)]['Annual'] += final_data_recording[str(test_beginyear+iyear)][imonth]
                        obs_data_recording[str(test_beginyear+iyear)]['Annual'] += obs_data_recording[str(test_beginyear+iyear)][imonth]
                        geo_data_recording[str(test_beginyear+iyear)]['Annual'] += geo_data_recording[str(test_beginyear+iyear)][imonth]
                        training_final_data_recording[str(test_beginyear+iyear)]['Annual'] += training_final_data_recording[str(test_beginyear+iyear)][imonth]
                        testing_population_data_recording[str(test_beginyear+iyear)]['Annual'] += testing_population_data_recording[str(test_beginyear+iyear)][imonth]
                        training_obs_data_recording[str(test_beginyear+iyear)]['Annual'] += training_obs_data_recording[str(test_beginyear+iyear)][imonth]
                        #PWAModel[str(test_beginyear+iyear)]['Annual']  += PWAModel[str(test_beginyear+iyear)][imonth]
                        #PWAMonitors[str(test_beginyear+iyear)]['Annual'] += PWAMonitors[str(test_beginyear+iyear)][imonth]
                    
                final_data_recording[str(test_beginyear+iyear)]['Annual'] = final_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
                obs_data_recording[str(test_beginyear+iyear)]['Annual'] = obs_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
                geo_data_recording[str(test_beginyear+iyear)]['Annual'] = geo_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
                training_final_data_recording[str(test_beginyear+iyear)]['Annual'] = training_final_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
                training_obs_data_recording[str(test_beginyear+iyear)]['Annual'] = training_obs_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
                testing_population_data_recording[str(test_beginyear+iyear)]['Annual'] = testing_population_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
                #PWAModel[str(test_beginyear+iyear)]['Annual'] = PWAModel[str(test_beginyear+iyear)]['Annual']/12.0
                #PWAMonitors[str(test_beginyear+iyear)]['Annual'] = PWAMonitors[str(test_beginyear+iyear)]['Annual']/12.0
                
                #### Get Annaul All Points Recording
                All_points_obs_data_recording['Annual']   = np.append(All_points_obs_data_recording['Annual'],obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                All_points_geo_data_recording['Annual']   = np.append(All_points_geo_data_recording['Annual'],geo_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                All_points_final_data_recording['Annual'] = np.append(All_points_final_data_recording['Annual'],final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                All_points_train_final_data_recording['Annual'] = np.append(All_points_train_final_data_recording['Annual'],training_final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                All_points_train_obs_data_recording['Annual'] = np.append(All_points_train_obs_data_recording['Annual'],training_obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                All_points_population_data_recording['Annual'] = np.append(All_points_population_data_recording['Annual'],testing_population_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])

                #### Calculate Annual Statistics for AVD Spatial CV
                print('Area: {}, Year: {}, Month: {}'.format('NA', test_beginyear+iyear, 'Annual'))
                test_CV_R2[str(test_beginyear+iyear)]['Annual'] = linear_regression(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                train_CV_R2[str(test_beginyear+iyear)]['Annual'] = linear_regression(training_final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], training_obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                geo_CV_R2[str(test_beginyear+iyear)]['Annual'] = linear_regression(geo_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                RMSE[str(test_beginyear+iyear)]['Annual'] = Cal_RMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                NRMSE[str(test_beginyear+iyear)]['Annual'] = Cal_NRMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                PWM_NRMSE[str(test_beginyear+iyear)]['Annual'] = Cal_PWM_rRMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],testing_population_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                
                PWAModel[str(test_beginyear+iyear)]['Annual'] = Calculate_PWA_PM25(testing_population_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                PWAMonitors [str(test_beginyear+iyear)]['Annual'] = Calculate_PWA_PM25(testing_population_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                regression_Dic = regress2(_x= obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],_y=final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
                slopes[str(test_beginyear+iyear)]['Annual'] = slope

                for iseason in range(len(Seasons)):
                    ##### Get Seasonal Data Recording
                    temp_final_data_recording = np.zeros(len(final_data_recording[str(test_beginyear+iyear)]['Jan']),dtype=np.float32)
                    temp_obs_data_recording   = np.zeros(len(obs_data_recording[str(test_beginyear+iyear)]['Jan']),dtype=np.float32)
                    temp_geo_data_recording   = np.zeros(len(geo_data_recording[str(test_beginyear+iyear)]['Jan']),dtype=np.float32)
                    temp_training_final_data_recording   = np.zeros(len(training_final_data_recording[str(test_beginyear+iyear)]['Jan']),dtype=np.float32)
                    temp_training_obs_data_recording   = np.zeros(len(training_obs_data_recording[str(test_beginyear+iyear)]['Jan']),dtype=np.float32)
                    temp_testing_population_data_recording   = np.zeros(len(testing_population_data_recording[str(test_beginyear+iyear)]['Jan']),dtype=np.float32)
                    for imonth in Season_MONTH[iseason]:
                        print('{} Season_Month[iseason]: {}'.format(iseason, imonth))
                        print('temp_obs_data_recording: ',temp_obs_data_recording)
                        print('obs_data_recording[str(test_beginyear+iyear)][imonth]/3.0: ',obs_data_recording[str(test_beginyear+iyear)][imonth]/3.0)
                        temp_final_data_recording += final_data_recording[str(test_beginyear+iyear)][imonth]/3.0
                        temp_obs_data_recording   += obs_data_recording[str(test_beginyear+iyear)][imonth]/3.0
                        temp_geo_data_recording   += geo_data_recording[str(test_beginyear+iyear)][imonth]/3.0
                        temp_training_final_data_recording += training_final_data_recording[str(test_beginyear+iyear)][imonth]/3.0
                        temp_training_obs_data_recording   += training_obs_data_recording[str(test_beginyear+iyear)][imonth]/3.0
                        temp_testing_population_data_recording += testing_population_data_recording[str(test_beginyear+iyear)][imonth]/3.0
                    ### Seasonal All Points Recording
                    All_points_obs_data_recording[Seasons[iseason]]   = np.append(All_points_obs_data_recording[Seasons[iseason]],temp_obs_data_recording[masked_array_index])
                    All_points_geo_data_recording[Seasons[iseason]]   = np.append(All_points_geo_data_recording[Seasons[iseason]],temp_geo_data_recording[masked_array_index])
                    All_points_final_data_recording[Seasons[iseason]] = np.append(All_points_final_data_recording[Seasons[iseason]],temp_final_data_recording[masked_array_index])
                    All_points_train_final_data_recording[Seasons[iseason]] = np.append(All_points_train_final_data_recording[Seasons[iseason]],temp_training_final_data_recording[masked_array_index])
                    All_points_train_obs_data_recording[Seasons[iseason]] = np.append(All_points_train_obs_data_recording[Seasons[iseason]],temp_training_obs_data_recording[masked_array_index])
                    All_points_population_data_recording[Seasons[iseason]] = np.append(All_points_population_data_recording[Seasons[iseason]],temp_testing_population_data_recording[masked_array_index])
                    ### Calculate Seasonal Statistics for AVD Spatial CV
                    print('Area: {}, Year: {}, Season: {}'.format('NA', test_beginyear+iyear, Seasons[iseason]))
                    test_CV_R2[str(test_beginyear+iyear)][Seasons[iseason]] = linear_regression(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    train_CV_R2[str(test_beginyear+iyear)][Seasons[iseason]] = linear_regression(temp_training_final_data_recording[masked_array_index], temp_training_obs_data_recording[masked_array_index])
                    geo_CV_R2[str(test_beginyear+iyear)][Seasons[iseason]] = linear_regression(temp_geo_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    RMSE[str(test_beginyear+iyear)][Seasons[iseason]] = Cal_RMSE(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    NRMSE[str(test_beginyear+iyear)][Seasons[iseason]] = Cal_NRMSE(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    PWM_NRMSE[str(test_beginyear+iyear)][Seasons[iseason]] = Cal_PWM_rRMSE(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index], temp_testing_population_data_recording[masked_array_index])
                    
                    regression_Dic = regress2(_x= temp_obs_data_recording,_y=temp_final_data_recording,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                    intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
                    slopes[str(test_beginyear+iyear)][Seasons[iseason]] = slope
                    PWAModel[str(test_beginyear+iyear)][Seasons[iseason]] = Calculate_PWA_PM25(Population_array=temp_testing_population_data_recording[masked_array_index],PM25_array=temp_final_data_recording[masked_array_index])
                    PWAMonitors[str(test_beginyear+iyear)][Seasons[iseason]] = Calculate_PWA_PM25(Population_array=temp_testing_population_data_recording[masked_array_index],PM25_array=temp_obs_data_recording[masked_array_index])
        ###### Calculate All Points Statistics
        AllPoints_TimePeriods = ['AllPoints','Annual','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','MAM','JJA','SON','DJF']
        for itime in AllPoints_TimePeriods:

            test_CV_R2['AllPoints'][itime]  = linear_regression(All_points_obs_data_recording[itime],All_points_final_data_recording[itime])
            train_CV_R2['AllPoints'][itime] = linear_regression(All_points_train_obs_data_recording[itime],All_points_train_final_data_recording[itime])
            geo_CV_R2['AllPoints'][itime]   = linear_regression(All_points_geo_data_recording[itime],All_points_obs_data_recording[itime])
            RMSE['AllPoints'][itime]        = Cal_RMSE(All_points_final_data_recording[itime],All_points_obs_data_recording[itime])
            NRMSE['AllPoints'][itime]       = Cal_NRMSE(All_points_final_data_recording[itime],All_points_obs_data_recording[itime])
            PWM_NRMSE['AllPoints'][itime]       = Cal_PWM_rRMSE(All_points_final_data_recording[itime],All_points_obs_data_recording[itime], All_points_population_data_recording[itime])
            regression_Dic = regress2(_x= All_points_obs_data_recording[itime],_y=All_points_final_data_recording[itime],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
            slopes['AllPoints'][itime] = slope
            PWAModel['AllPoints'][itime] = Calculate_PWA_PM25(Population_array=All_points_population_data_recording[itime],PM25_array=All_points_final_data_recording[itime])
            PWAMonitors['AllPoints'][itime] = Calculate_PWA_PM25(Population_array=All_points_population_data_recording[itime],PM25_array=All_points_obs_data_recording[itime])
    else:
        for iyear in range(test_endyear-test_beginyear+1):
                for imonth in MONTH:
                    print('Area: {}, Year: {}, Month: {}'.format('NA', test_beginyear+iyear, imonth))
                    test_CV_R2[str(test_beginyear+iyear)][imonth] = 0
                    train_CV_R2[str(test_beginyear+iyear)][imonth] = 0#linear_regression(training_final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], training_obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    geo_CV_R2[str(test_beginyear+iyear)][imonth] = 0#linear_regression(geo_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    RMSE[str(test_beginyear+iyear)][imonth] = 0#Cal_RMSE(final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    PWM_NRMSE[str(test_beginyear+iyear)][imonth] = 0#Cal_PWM_rRMSE(final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index],testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    NRMSE[str(test_beginyear+iyear)][imonth] = 0#Cal_NRMSE(final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index], obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])

                    PWAModel[str(test_beginyear+iyear)][imonth] = 0#Calculate_PWA_PM25(Population_array=testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index],PM25_array=final_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                    PWAMonitors[str(test_beginyear+iyear)][imonth] = 0#Calculate_PWA_PM25(Population_array=testing_population_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index],PM25_array=obs_data_recording[str(test_beginyear+iyear)][imonth][masked_array_index])
                        
                test_CV_R2[str(test_beginyear+iyear)]['Annual'] = 0#linear_regression(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                train_CV_R2[str(test_beginyear+iyear)]['Annual'] = 0#linear_regression(training_final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], training_obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                geo_CV_R2[str(test_beginyear+iyear)]['Annual'] = 0#linear_regression(geo_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                RMSE[str(test_beginyear+iyear)]['Annual'] =0# Cal_RMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                NRMSE[str(test_beginyear+iyear)]['Annual'] = 0#Cal_NRMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                PWM_NRMSE[str(test_beginyear+iyear)]['Annual'] = 0#Cal_PWM_rRMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index], obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],testing_population_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                
                PWAModel[str(test_beginyear+iyear)]['Annual'] = 0#Calculate_PWA_PM25(testing_population_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],final_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                PWAMonitors [str(test_beginyear+iyear)]['Annual'] = 0#Calculate_PWA_PM25(testing_population_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index],obs_data_recording[str(test_beginyear+iyear)]['Annual'][masked_array_index])
                for iseason in range(len(Seasons)):
                    test_CV_R2[str(test_beginyear+iyear)][Seasons[iseason]] = 0#linear_regression(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    train_CV_R2[str(test_beginyear+iyear)][Seasons[iseason]] = 0#linear_regression(temp_training_final_data_recording[masked_array_index], temp_training_obs_data_recording[masked_array_index])
                    geo_CV_R2[str(test_beginyear+iyear)][Seasons[iseason]] = 0#linear_regression(temp_geo_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    RMSE[str(test_beginyear+iyear)][Seasons[iseason]] = 0#Cal_RMSE(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    NRMSE[str(test_beginyear+iyear)][Seasons[iseason]] = 0#Cal_NRMSE(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index])
                    PWM_NRMSE[str(test_beginyear+iyear)][Seasons[iseason]] = 0#Cal_PWM_rRMSE(temp_final_data_recording[masked_array_index], temp_obs_data_recording[masked_array_index], temp_testing_population_data_recording[masked_array_index])
                    PWAModel[str(test_beginyear+iyear)][Seasons[iseason]] = 0#Calculate_PWA_PM25(Population_array=temp_testing_population_data_recording[masked_array_index],PM25_array=temp_final_data_recording[masked_array_index])
                    PWAMonitors[str(test_beginyear+iyear)][Seasons[iseason]] = 0#Calculate_PWA_PM25(Population_array=temp_testing_population_data_recording[masked_array_index],PM25_array=temp_obs_data_recording[masked_array_index])
        AllPoints_TimePeriods = ['AllPoints','Annual','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','MAM','JJA','SON','DJF']
        for itime in AllPoints_TimePeriods:

            test_CV_R2['AllPoints'][itime]  = 0#linear_regression(All_points_obs_data_recording[itime],All_points_final_data_recording[itime])
            train_CV_R2['AllPoints'][itime] = 0#linear_regression(All_points_train_obs_data_recording[itime],All_points_train_final_data_recording[itime])
            geo_CV_R2['AllPoints'][itime]   = 0#linear_regression(All_points_geo_data_recording[itime],All_points_obs_data_recording[itime])
            RMSE['AllPoints'][itime]        = 0#Cal_RMSE(All_points_final_data_recording[itime],All_points_obs_data_recording[itime])
            NRMSE['AllPoints'][itime]       = 0#Cal_NRMSE(All_points_final_data_recording[itime],All_points_obs_data_recording[itime])
            #regression_Dic = regress2(_x= All_points_obs_data_recording[itime],_y=All_points_final_data_recording[itime],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            #intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
            slopes['AllPoints'][itime] = 0#slope
            PWAModel['AllPoints'][itime] = 0#Calculate_PWA_PM25(Population_array=All_points_population_data_recording[itime],PM25_array=All_points_final_data_recording[itime])
            PWAMonitors['AllPoints'][itime] = 0#Calculate_PWA_PM25(Population_array=All_points_population_data_recording[itime],PM25_array=All_points_obs_data_recording[itime])
                
    
    return test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slopes, PWAModel, PWAMonitors, len(masked_array_index)


def calculate_Alltime_Statistics_results(test_beginyear:dict,test_endyear:int,test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE,PWM_NRMSE, slope,PWAModel,PWAMonitors ):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','Annual','MAM','JJA','SON','DJF']
    test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_Alltime, NRMSE_Alltime, PWM_NRMSE_Alltime, slope_Alltime,PWAModel_Alltime, PWAMonitors_Alltime = initialize_AVD_CV_Alltime_dict()
    
    for imonth in MONTH:
            temp_test_CV_R2_Alltime   = np.array([],dtype=np.float64)
            temp_train_CV_R2_Alltime  = np.array([],dtype=np.float64)
            temp_geo_CV_R2_Alltime    = np.array([],dtype=np.float64)
            temp_RMSE_Alltime         = np.array([],dtype=np.float64)
            temp_NRMSE_Alltime        = np.array([],dtype=np.float64)
            temp_PWM_NRMSE_Alltime    = np.array([],dtype=np.float64)
            temp_slope_Alltime        = np.array([],dtype=np.float64)
            temp_PWAModel_Alltime     = np.array([],dtype=np.float64)
            temp_PWAMonitors_Alltime  = np.array([],dtype=np.float64)
            for iyear in range(test_endyear-test_beginyear+1):
                print('Area: {}, Year: {}, Month: {}'.format('NA', test_beginyear+iyear, imonth))
                temp_test_CV_R2_Alltime  = np.append(temp_test_CV_R2_Alltime, test_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_train_CV_R2_Alltime = np.append(temp_train_CV_R2_Alltime, train_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_geo_CV_R2_Alltime   = np.append(temp_geo_CV_R2_Alltime, geo_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_RMSE_Alltime        = np.append(temp_RMSE_Alltime, RMSE[str(test_beginyear+iyear)][imonth])
                temp_NRMSE_Alltime       = np.append(temp_NRMSE_Alltime, NRMSE[str(test_beginyear+iyear)][imonth])
                temp_PWM_NRMSE_Alltime   = np.append(temp_PWM_NRMSE_Alltime, PWM_NRMSE[str(test_beginyear+iyear)][imonth])
                temp_slope_Alltime       = np.append(temp_slope_Alltime, slope[str(test_beginyear+iyear)][imonth])
                temp_PWAModel_Alltime    = np.append(temp_PWAModel_Alltime, PWAModel[str(test_beginyear+iyear)][imonth])
                temp_PWAMonitors_Alltime = np.append(temp_PWAMonitors_Alltime, PWAMonitors[str(test_beginyear+iyear)][imonth])
            
            test_CV_R2_Alltime['Alltime'][imonth]     = get_mean_min_max_statistic(temp_test_CV_R2_Alltime)
            train_CV_R2_Alltime['Alltime'][imonth]    = get_mean_min_max_statistic(temp_train_CV_R2_Alltime)
            geo_CV_R2_Alltime['Alltime'][imonth]      = get_mean_min_max_statistic(temp_geo_CV_R2_Alltime)
            RMSE_Alltime['Alltime'][imonth]           = get_mean_min_max_statistic(temp_RMSE_Alltime)
            NRMSE_Alltime['Alltime'][imonth]          = get_mean_min_max_statistic(temp_NRMSE_Alltime)
            PWM_NRMSE_Alltime['Alltime'][imonth]      = get_mean_min_max_statistic(temp_PWM_NRMSE_Alltime)
            slope_Alltime['Alltime'][imonth]          = get_mean_min_max_statistic(temp_slope_Alltime)
            PWAModel_Alltime['Alltime'][imonth]       = get_mean_min_max_statistic(temp_PWAModel_Alltime)
            PWAMonitors_Alltime['Alltime'][imonth]    = get_mean_min_max_statistic(temp_PWAMonitors_Alltime)

    return test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime,RMSE_Alltime, NRMSE_Alltime, PWM_NRMSE_Alltime,slope_Alltime,PWAModel_Alltime,PWAMonitors_Alltime

def get_mean_min_max_statistic(temp_CV):
    temp_array = np.zeros((4),dtype=np.float64)
    temp_array[0] = np.mean(temp_CV)
    temp_array[1] = np.min(temp_CV)
    temp_array[2] = np.max(temp_CV)
    temp_array[3] = np.std(temp_CV)
    return temp_array

def CalculateAnnualR2(test_index,final_data,test_obs_data,beginyear,endyear):
    '''
    This funciton is used to calculate the Annual R2, slope and RMSE
    return:
    annual_R2,annual_final_data,annual_mean_obs,slope, RMSE
    '''
    annual_mean_obs = np.zeros((len(test_index)))
    annual_final_data = np.zeros((len(test_index)))
    test_month = np.array(range((endyear - beginyear + 1) * 12))
    for isite in range(len(test_index)):
        annual_mean_obs[isite] = np.mean(test_obs_data[isite + test_month * len(test_index)])
        annual_final_data[isite] = np.mean(final_data[isite + test_month * len(test_index)])
    print(' ################### Annual R2: #######################')
    annual_R2 = linear_regression(annual_mean_obs, annual_final_data)
    regression_Dic = regress2(_x=annual_mean_obs,_y=annual_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',
    )
    intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    intercept = round(intercept, 2)
    slope = round(slope, 2)
    RMSE = Cal_RMSE(annual_mean_obs, annual_final_data)
    return annual_R2,annual_final_data,annual_mean_obs,slope,RMSE

def CalculateMonthR2(test_index,final_data,test_obs_data,beginyear:int,endyear:int,monthly_final_test_imodel,monthly_obs_test_imodel):
    '''
    This funciton is used to calculate the monthly R2, slope and RMSE
    return:
    month_R2, month_slope, month_RMSE
    '''
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    month_obs = np.zeros((len(test_index)))
    month_predict = np.zeros((len(test_index)))

    monthly_test_month = np.array(range(endyear - beginyear + 1)) * 12
    month_R2 = np.zeros(12,dtype = np.float64)
    month_slope = np.zeros(12,dtype = np.float64)
    month_RMSE = np.zeros(12,dtype = np.float64)
   

    for imonth in range(12):
        for isite in range(len(test_index)):
            month_obs[isite] = np.mean(test_obs_data[isite + (imonth + monthly_test_month) * len(test_index)])
            month_predict[isite] = np.mean(final_data[isite + (imonth + monthly_test_month) * len(test_index)])
        monthly_final_test_imodel[MONTH[imonth]] = np.append(monthly_final_test_imodel[MONTH[imonth]], month_predict)
        monthly_obs_test_imodel[MONTH[imonth]]   = np.append(monthly_obs_test_imodel[MONTH[imonth]], month_obs)
        print('-------------------- Month: {} --------------------------'.format(MONTH[imonth]))
        month_R2[imonth] = linear_regression(month_obs, month_predict)
        regression_Dic = regress2(_x=month_obs,_y=month_predict,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
        intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
        month_slope[imonth] = round(slope, 2)
        month_RMSE[imonth] = Cal_RMSE(month_obs, month_predict)
        
    return month_R2, month_slope, month_RMSE, monthly_final_test_imodel, monthly_obs_test_imodel

