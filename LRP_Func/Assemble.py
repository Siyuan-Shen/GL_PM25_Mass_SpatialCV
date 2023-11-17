import numpy as np
import torch
from .lrp import LRPModel,LRPResModel
import csv
import os
from sklearn.model_selection import RepeatedKFold
from .visualize import plot_importance,plot_relevance_scores,sort_score_list
from Spatial_CV.Net_Construction import ResNet
from Spatial_CV.ConvNet_Data import Normlize_Training_Datasets
from Spatial_CV.utils import get_area_index





def MultiyearAreaModelLRP(train_input: torch.tensor, true_input:torch.tensor,
                        channel_index:np.array, kfold:int, repeats:int,
                         extent:np.array,
                         Area:str,version:str,special_name:str,model_outdir:str,
                         databeginyear:int,beginyear:np.array, endyear:np.array, bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,
                         Log_PM25:bool,WindSpeed_Abs:bool,calculate:bool,plot:bool):
    init_channel_name = ['EtaAOD_Bias','EtaCoastal','EtaMixing','EtaSGAOD_Bias','EtaSGTOPO_Bias',
                'AOD','ETA',
                'GC_PM25','GC_NH4','GC_SO4','GC_NIT','GC_SOA','GC_OC','GC_BC','GC_DST','GC_SSLT',
                'GeoPM25','Lat','Lon',
                'Landtype','Elevation',
                'BC_Emi','OC_Emi','DST_Emi','SSLT_Emi',
                'PBLH','T2M','V10M','U10M','RH',
                'Population',
                'SitesNumber']
    
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
    CV_R2, annual_CV_R2, month_CV_R2 = Initialize_CV_Dic(kfold=kfold,repeats=repeats,breakpoints=beginyear)
    area_index = get_area_index(extent=extent, test_index=site_index)
    channel_name = []
    for i in range(nchannel):
        channel_name.append(init_channel_name[channel_index[i]])
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
    annual_final_dic, annual_obs_dic = Initialize_DataRecording_Dic(breakpoints=beginyear)
    train_input, train_mean, train_std = Normlize_Training_Datasets(train_input,channel_index,Met_Absolute=WindSpeed_Abs)
    train_input = train_input[:,channel_index,:,:]
    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    relevance_outdir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/GlobalTraining_MultipleModel_Spatial_withAreas_Cross_Validation_BenchMark/data_output/Relevance_Score/' + Area +'/'
    if not os.path.isdir(relevance_outdir):
        os.makedirs(relevance_outdir)
    
    if calculate == True:
        for train_index, test_index in rkf.split(area_index):
        # *------------------------------------------------------------------------------*#
        ## Initialize the results arraies.
        ## For recording all models results in this fold.
        # *------------------------------------------------------------------------------*#
            for imodel in range(len(beginyear)):
                modelfile = model_outdir +'CNN_PM25_Spatial_'+typeName+'_'+Area+'_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_No' + str(count) + '.pt'
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cnn_model = torch.load(modelfile,map_location=torch.device(device)).eval()
                print('Model: ', modelfile,'\n has been loaded')
                X_index, Y_index = GetXYIndex(area_index=area_index,train_index=train_index,test_index=test_index,
                                          beginyear=beginyear[imodel],endyear=endyear[imodel],databeginyear=databeginyear,
                                          GLsitesNum=len(site_index))
                X_train, X_test = train_input[X_index, :, :, :], true_input[X_index]
                y_train, y_test = train_input[Y_index, :, :, :], true_input[Y_index]

            
            # *------------------------------------------------------------------------------*#
            ## Training Process.
            # *------------------------------------------------------------------------------*#
    
                print('y train shape:', y_train.shape)

                cnn_model.to(device)## to device
                lrp_model = LRPResModel(cnn_model)
                torch.manual_seed(21)
                y_tensor = torch.Tensor(y_train).to(device)
                print('y_tensor.is_leaf:',y_tensor.is_leaf)
                #y_train.to(device)## to device
                temp_relevance_score = lrp_model.forward(y_tensor)
                if count == 0:
                    relevance_score = temp_relevance_score
                else:
                    relevance_score = np.append(relevance_score,temp_relevance_score,axis = 0)
            count += 1

        # *------------------------------------------------------------------------------*#
        ## Calculate the correlation R2 for all models this fold
        # *------------------------------------------------------------------------------*#
        relevance_score_outfile = relevance_outdir + 'Spatial_CV_'+typeName+'_v' + version + '_' + str(nchannel) + 'Channel_'+Area+'_' + str(width) + 'x' + str(width) + special_name + '.npy'

        np.save(relevance_score_outfile,relevance_score)
    
    if plot == True:
        relevance_score_infile = relevance_outdir + 'Spatial_CV_'+typeName+'_v' + version + '_' + str(nchannel) + 'Channel_'+Area+'_' + str(width) + 'x' + str(width) + special_name + '.npy'
        relevance_score = np.abs(np.load(relevance_score_infile))
        avg_relevance_score = np.mean(relevance_score, axis=0)
        print(avg_relevance_score.shape)

    ### Plot the relevance score heatmap
        figure_outdir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/GlobalTraining_MultipleModel_Spatial_withAreas_Cross_Validation_BenchMark/figures/'
    
        for i in range(nchannel):
            relevance_heatmat_outdir = figure_outdir+'Relevance_Scores_Heatmap/v'+version+'/'
            if not os.path.isdir(relevance_heatmat_outdir):
                os.makedirs(relevance_heatmat_outdir)
            relevance_heatmap_outfile = relevance_heatmat_outdir + channel_name[i]+'CNN_'+typeName+'_Spatial_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name + '_R_Heatmap.png'
            plot_relevance_scores(avg_relevance_score[i,:,:],relevance_heatmap_outfile) 
        
        ### Get the importance score for each channel
        importance_score = np.sum(avg_relevance_score, axis=(1,2))
        sum_importance_score = np.sum(importance_score)
        print(sum_importance_score)
        importance_score = importance_score/sum_importance_score

        ### Plot the channel importance list
        sorted_r,sorted_namelist = sort_score_list(importance_score,channel_name,MaxToMin = False)
        importance_outdir = figure_outdir+ 'Importance_List/v'+version+'/'
        if not os.path.isdir(importance_outdir):
                os.makedirs(importance_outdir)
        importance_outfile = importance_outdir + 'CNN_'+typeName+'_Spatial_NoAbs_2022' + version + '_' + str(
                nchannel) + 'Channel' + special_name +  '.png'
        plot_importance(sorted_r,sorted_namelist,importance_outfile)
        print(importance_score)
        

    
    


def Output_Text(outfile:str,status:str,CV_R2:np.array,annual_CV_R2:np.array,month_CV_R2:np.array,beginyear:str,
                endyear:str,Area:str,kfold:int,repeats:int):
    MONTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    CV_R2[-1] = np.mean(CV_R2[0:kfold * repeats])
    annual_CV_R2[-1] = np.mean(annual_CV_R2[0:kfold * repeats])


    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Area,'Area ','Time Period: ', beginyear,' ', endyear])
        writer.writerow(['R2 for monthly validation','Max: ',str(np.round(np.max(CV_R2),4)),'Min: ',str(np.round(np.min(CV_R2),4)),
                         'Avg: ',str(np.round(CV_R2[-1],4))])
        writer.writerow(['R2 for Annual average validation', 'Max: ', str(np.round(np.max(annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_R2), 4)),
                         'Avg: ', str(np.round(annual_CV_R2[-1], 4))])
        for imonth in range(len(MONTH)):
            month_CV_R2[imonth,-1] = np.mean(month_CV_R2[imonth,0:kfold * repeats])
            writer.writerow(['R2 for ', MONTH[imonth], 'Max: ', str(np.round(np.max(month_CV_R2[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(month_CV_R2[imonth,-1],4))])



    return


def Data_Augmentation(X_train:np.array,X_test:np.array,Tranpose_Augmentation:bool,Flip_Augmentation:bool):
    X_train_output = X_train
    X_test_output  = X_test
    if Tranpose_Augmentation == True:
        X_train_Trans,X_test_Trans = TransposeTrainingData(X_train=X_train,X_test=X_test)
        X_train_output = np.append(X_train_output,X_train_Trans,axis = 0)
        X_test_output = np.append(X_test_output,X_test_Trans)
    if Flip_Augmentation == True:
        X_train_flip0,X_train_flip1,X_test_flip0,X_test_flip1 = FlipTraingData(X_train=X_train,X_test=X_test)
        X_train_output = np.append(X_train_output,X_train_flip0,axis = 0)
        X_train_output = np.append(X_train_output,X_train_flip1,axis = 0)
        X_test_output  = np.append(X_test_output,X_test_flip0)
        X_test_output  = np.append(X_test_output,X_test_flip1)
    return X_train_output,X_test_output

def FlipTraingData(X_train:np.array,X_test:np.array):
    X_train_flip0 = np.flip(X_train,2)
    X_train_flip1 = np.flip(X_train,3)
    X_test_flip0 = X_test
    X_test_flip1 = X_test
    return X_train_flip0,X_train_flip1,X_test_flip0,X_test_flip1

def TransposeTrainingData(X_train:np.array,X_test:np.array):
    X_train_Trans = X_train.transpose(0, 1, 3, 2)
    #X_train_double[0:len(X_index), :, :, :] = X_train
    #X_train_double[len(X_index):2 * len(X_index), :, :, :] = X_train_Trans
    X_test_Trans = X_test
    return  X_train_Trans,X_test_Trans

def GetXYIndex(area_index:np.array,train_index:np.array, test_index:np.array, beginyear:int, endyear:int,
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
    Y_index = np.zeros((12 * (endyear - beginyear + 1) * len(test_index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        X_index[i * len(train_index):(i + 1) * len(train_index)] = ((beginyear - databeginyear) * 12 + i) * GLsitesNum + \
                                                                   area_index[train_index]
        Y_index[i * len(test_index):(i + 1) * len(test_index)] = ((beginyear - databeginyear) * 12 + i) * GLsitesNum + \
                                                                 area_index[test_index]
    return X_index,Y_index

def Initialize_DataRecording_Dic(breakpoints:np.array):
    annual_final_test = {}
    annual_obs_test   = {}
    for imodel in range(len(breakpoints)):
        annual_final_test[str(breakpoints[imodel])] = np.array([],dtype=np.float64)
        annual_obs_test[str(breakpoints[imodel])] = np.array([],dtype=np.float64)

    annual_final_test['Alltime'] =  np.array([],dtype=np.float64)
    annual_obs_test['Alltime']   =  np.array([],dtype=np.float64)
    return annual_final_test,annual_obs_test

def Initialize_CV_Dic(kfold:int, repeats:int, breakpoints:np.array):
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
    annual_CV_R2 = {}
    month_CV_R2 = {}
    for imodel in range(len(breakpoints)):
        CV_R2[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_R2[str(breakpoints[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        month_CV_R2[str(breakpoints[imodel])] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    month_CV_R2['Alltime'] = np.zeros((12, kfold * repeats + 1), dtype=np.float32)
    return CV_R2, annual_CV_R2, month_CV_R2

def Initialize_CV_array(kfold:int, repeats:int):
    '''
    The function is to initialize the CV recording arrays
    :param kfold: k number of folds
    :param repeats: repeat time
    :return: CV_R2 - record the R2 for each fold(original data).
             annual_CV_R2 - record the R2 for purely spatial R2.
             month_CV_R2  - record the R2 for purely spatial R2 for months.
    '''
    CV_R2 = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_R2 = np.zeros((kfold * repeats + 1), dtype=np.float32)
    month_CV_R2 = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    return CV_R2, annual_CV_R2, month_CV_R2

def Get_data_NormPara(input_dir:str,input_file:str):
    infile = input_dir + input_file
    data   = np.load(infile)
    data_mean = np.mean(data)
    data_std  = np.std(data)
    return data, data_mean, data_std

def Get_CV_seed():
    seed = 19980130
    print('Seed is :', seed)
    return seed


