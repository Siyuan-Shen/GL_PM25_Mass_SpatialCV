import csv
import numpy as np
import time
import gc
from Spatial_CV.CV_Func import MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25_AllfoldsTogether_GBDAreas,MultiyearAreaModelCrossValid,plot_from_data, MultiyearMultiAreasSpatialCrossValidation, EachAreaForcedSlope_MultiyearMultiAreasSpatialCrossValidation, MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25, MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25_GBDAreas
from Spatial_CV.ConvNet_Data import Learning_Object_Datasets
from Spatial_CV.utils import extent_table
from LRP_Func.Assemble import MultiyearAreaModelLRP




#######################################################################################
##                               Cross Validation SOP                                ##
## 1. Change Net.                                                                    ##
##    -> change the net in Saptial_CV.Net_Construction.py to desired Net.            ##
##    -> change the import parts in Spatial_CV.CV_Func, and the net line in Func     ##
##    {MultiyearAreaModelCrossValid}                                                 ##
##    -> check the special_name.                                                     ##
##                                                                                     
#######################################################################################



total_time_start = time.time()
#######################################################################################
##                                   Initial Settings                                ##
#######################################################################################
YYYY = ['1998', '1999', '2000', '2001', '2002', '2003', '2004',
        '2005', '2006', '2007', '2008', '2009', '2010', '2011',
        '2012', '2013', '2014', '2015', '2016', '2017', '2018',
        '2019']
MM = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
channel_name = ['EtaAOD_Bias','EtaCoastal','EtaMixing','EtaSGAOD_Bias','EtaSGTOPO_Bias',
                'AOD','ETA',
                'GC_PM25','GC_NH4','GC_SO4','GC_NIT','GC_SOA','GC_OC','GC_BC','GC_DST','GC_SSLT',
                'GeoPM25','Lat','Lon',
                'Landtype','Elevation',
                'BC_Emi','OC_Emi','DST_Emi','SSLT_Emi',
                'PBLH','T2M','V10M','U10M','RH',
                'Population',
                'SitesNumber',
                'GFED4_TOTL_DM_Emi']

'''
channel_name = ['EtaAOD_Bias','EtaCoastal','EtaMixing','EtaSGAOD_Bias','EtaSGTOPO_Bias',
                'AOD','ETA',
                'GC_PM25','GC_NH4','GC_SO4','GC_NIT','GC_SOA','GC_OC','GC_BC','GC_DST','GC_SSLT',
                'GeoPM25','Lat','Lon',
                'Landtype','Elevation',
                'BC_Emi','OC_Emi','DST_Emi','SSLT_Emi',
                'PBLH','T2M','V10M','U10M','RH', 'GRN', 'PRECTOT','Q850','PS'
                'Population',
                'Total_RoadDensity','Type1_RoadDensity','Type2_RoadDensity','Type3_RoadDensity','Type4_RoadDensity','Type5_RoadDensity'
                ]
'''
'''
channel_name = ['EtaAOD_Bias','EtaCoastal','EtaMixing','EtaSGAOD_Bias','EtaSGTOPO_Bias',
                'AOD','ETA',
                'GC_PM25','GC_NH4','GC_SO4','GC_NIT','GC_SOA','GC_OC','GC_BC','GC_DST','GC_SSLT',
                'GeoPM25','Lat','Lon',
                'Landtype','Elevation',
                'BC_Emi','OC_Emi','DST_Emi','SSLT_Emi',
                'PBLH','T2M','V10M','U10M','RH',
                'Population',
                'Total_RoadDensity','Type1_RoadDensity','Type2_RoadDensity','Type3_RoadDensity','Type4_RoadDensity','Type5_RoadDensity'
                ]
'''
#channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#channel_index = [29,9,28,8,23,24,15,18,27,4,5,17,12,11,1,2,16,26,13]
#channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,31]
# channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,33] #Met Extra
#normlized_channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21,22,23,24,25,26,27,28,29]
#channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29]
channel_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29]
nchannel = len(channel_index)


#######################################################################################
##                             Input and output Directories                          ##
#######################################################################################

input_dir = '/my-projects/Projects/MLCNN_PM25_2021/data/'
# train_infile = input_dir + 'CNN_Training_data_NCWH_11x11_channel31_start1998.npy'
#train_infile = input_dir + 'CNN_Training_data_NCWH_11x11_channel31_start1998_OnlyInterpolateGCVariables.npy'
#train_infile = input_dir + 'CNN_Training_data_NCWH_11x11_channel31_start1998_InterpolateVariables.npy'
#train_infile = input_dir + 'CNN_Training_data_NCWH_11x11_channel32_start1998_InterpolateVariables_200kmSitesNumber.npy'
train_infile = input_dir + 'CNN_Training_data_NCWH_11x11_channel33_start1998_InterpolateVariables_GFED4EMI.npy'
# train_infile = input_dir + 'CNN_Training_data_NCWH_11x11_channel31_start1998_InterpolateVariables_CountryMaskedPop.npy'

GeoPM25Enhenced_Training_infile = input_dir + 'CNN_Pretrained_data_NCWH_11x11_channel31_start2015.npy'
GeoPM25Enhenced_TrueValue_infile= input_dir + 'Pretrained_GeoPM25_start2015.npy'

#train_infile = input_dir + 'CNN_Training_data_NCWH_11x11_channel37_start2015_InterpolateVariables_CountryMaskedPop_RoadDensity.npy'
model_outdir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/GlobalTraining_MultipleModel_Spatial_withAreas_Cross_Validation_BenchMark/model_results/'

#######################################################################################
##                                Initial Arrays and Constants                       ##
#######################################################################################

kfold = 10
repeats = 1

num_epochs = 30
batchsize = 512
learning_rate = 0.01

#beginyear = [2001,2005,2010,2015]
#endyear = [2004,2009,2014,2019]
beginyear = [2015]
endyear = [2019]
databeginyear = 1998
GeoPM25Enforced_beginyear = 2015
version = 'v20231117'
Area = 'GL'
special_name = '_optimal_model_2015-2019_EachYear_lr0d01_bs512_epoch30'#_SigmoidMSELossWithGeoPenalties_alpha0d005_beta8d0_gamma3d0_lambda1-0d2' #'_exclude_longitude_landtype_GeoPenaltySum_constrain_alpha0d75_beta0d75_lambda1_0d5_lambda2_0d5'
extent_dic = extent_table()
extent = extent_dic[Area]


#########################################################
#                Data Augmentation Settings             #
#########################################################
augmentation = False
Transpose_augmentation = False
Flip_augmentation = False
AddNoise_Augmentation = False


#########################################################
#                 Variables Settings                    #
#########################################################
GeoPM25Enhenced = False ### Useless - No need anymore
WindSpeed_Abs = False   ### Set the wind speed to absolute value

#########################################################
#                   Main Process Settings               #
#########################################################
MultiAreas = True
CV = True
OnlyCV_plot = True

ForcedSlopeUnity = True # True: force the slope to unity and offset to zero with Training datasets
EachAreaForcedSlopeUnity = False # True: force the slope to unity and offset to zero by each area; False: by global
EachMonthForcedSlopeUnity = True # True: force the slope to unity and offset to zero by each year, each month; False: by each month, all year average 

Combine_with_GeoPM25 = True   #### For optimal model

LRP = False
LRP_Calculation = False
LRP_Plot =True




if __name__ == '__main__':
    bias = True
    Normalized_PM25 = False
    Absolute_PM25 = False
    Log_PM25 = False

    print('Train infile:',train_infile,'\nEpoch: ', num_epochs,'\n batchsize: ',batchsize,'\ninitial learning rate: ',learning_rate,
    '\nbeginyear: ', beginyear,'\nendyear: ',endyear,'\nversion:', version,'\nArea:', Area,'\nSpecial Name:', special_name,
    '\naugmentation:',augmentation,'\nTranspose Augmentation:',Transpose_augmentation,'\nFlip Augmentation: ',Flip_augmentation,
    '\nAdd Noise Augmentation:', AddNoise_Augmentation,'\nbias:',bias,'\nNormalized PM2.5: ',Normalized_PM25,'\nAbsolute PM2.5:', Absolute_PM25,'\nLog PM2.5: ',Log_PM25,
    '\nCV:',CV,'\nLRP:',LRP,'\nLRP Calculation:', LRP_Calculation,'\nLRP Plot:', LRP_Plot,'\nGeoPM25Enhenced: ',GeoPM25Enhenced,
    '\nWind Speed Absolute: ', WindSpeed_Abs,'\n Channel INDEX: ', channel_index,'\nChannel Name: ', channel_name,
    '\nForcedSlopeUnity: ',ForcedSlopeUnity, '\nEachAreaForcedSlopeUnity:',EachAreaForcedSlopeUnity,'\nEachMonthForcedSlopeUnity:',EachMonthForcedSlopeUnity)

    CV_time_start = time.time()
    if CV == True:
        train_input = np.load(train_infile)
        true_input = Learning_Object_Datasets(bias=bias,Normlized_PM25=Normalized_PM25,Absolute_PM25=Absolute_PM25,Log_PM25=Log_PM25)

        if MultiAreas == False:
            MultiyearAreaModelCrossValid(train_input=train_input,true_input=true_input,channel_index=channel_index,
    kfold=kfold,repeats=repeats,extent=extent,num_epochs=num_epochs,batch_size=batchsize,learning_rate=learning_rate,
    Area=Area,version=version,special_name=special_name,model_outdir=model_outdir,databeginyear=databeginyear,
    beginyear=beginyear,endyear=endyear,augmentation=augmentation,Tranpose_Augmentation=Transpose_augmentation,Flip_Augmentation=Flip_augmentation,AddNoise_Augmentation=AddNoise_Augmentation,
    bias=bias,Normlized_PM25=Normalized_PM25,Absolute_Pm25=Absolute_PM25,Log_PM25=Log_PM25,WindSpeed_Abs=WindSpeed_Abs,GeophysicalPM25Enhenced=GeoPM25Enhenced,
    GeoPM25Enhenced_Train_infile=GeoPM25Enhenced_Training_infile,GeoPM25Enhenced_True_infile=GeoPM25Enhenced_TrueValue_infile,
    GeoPM25Enhenced_DataStartYear=GeoPM25Enforced_beginyear)
        else:
            if ForcedSlopeUnity:
                if Combine_with_GeoPM25 == True:
                    txt_outfile = MultiyearMultiAreasBLOOSpatialCrossValidation_CombineWithGeophysicalPM25(train_input=train_input,true_input=true_input,channel_index=channel_index,
    kfold=kfold,repeats=repeats,extent=extent,num_epochs=num_epochs,batch_size=batchsize,learning_rate=learning_rate,
    Area=Area,version=version,special_name=special_name,model_outdir=model_outdir,databeginyear=databeginyear,
    beginyear=beginyear,endyear=endyear,augmentation=augmentation,Tranpose_Augmentation=Transpose_augmentation,Flip_Augmentation=Flip_augmentation,AddNoise_Augmentation=AddNoise_Augmentation,
    bias=bias,Normlized_PM25=Normalized_PM25,Absolute_Pm25=Absolute_PM25,Log_PM25=Log_PM25,WindSpeed_Abs=WindSpeed_Abs,GeophysicalPM25Enhenced=GeoPM25Enhenced,
    GeoPM25Enhenced_Train_infile=GeoPM25Enhenced_Training_infile,GeoPM25Enhenced_True_infile=GeoPM25Enhenced_TrueValue_infile,
    GeoPM25Enhenced_DataStartYear=GeoPM25Enforced_beginyear,EachMonthSlopeUnity=EachMonthForcedSlopeUnity,EachAreaForcedSlopeUnity=EachAreaForcedSlopeUnity)
                else:
                    txt_outfile = EachAreaForcedSlope_MultiyearMultiAreasSpatialCrossValidation(train_input=train_input,true_input=true_input,channel_index=channel_index,
    kfold=kfold,repeats=repeats,extent=extent,num_epochs=num_epochs,batch_size=batchsize,learning_rate=learning_rate,
    Area=Area,version=version,special_name=special_name,model_outdir=model_outdir,databeginyear=databeginyear,
    beginyear=beginyear,endyear=endyear,augmentation=augmentation,Tranpose_Augmentation=Transpose_augmentation,Flip_Augmentation=Flip_augmentation,AddNoise_Augmentation=AddNoise_Augmentation,
    bias=bias,Normlized_PM25=Normalized_PM25,Absolute_Pm25=Absolute_PM25,Log_PM25=Log_PM25,WindSpeed_Abs=WindSpeed_Abs,GeophysicalPM25Enhenced=GeoPM25Enhenced,
    GeoPM25Enhenced_Train_infile=GeoPM25Enhenced_Training_infile,GeoPM25Enhenced_True_infile=GeoPM25Enhenced_TrueValue_infile,
    GeoPM25Enhenced_DataStartYear=GeoPM25Enforced_beginyear,EachMonthSlopeUnity=EachMonthForcedSlopeUnity,EachAreaForcedSlopeUnity=EachAreaForcedSlopeUnity)
            else:
                txt_outfile = MultiyearMultiAreasSpatialCrossValidation(train_input=train_input,true_input=true_input,channel_index=channel_index,
    kfold=kfold,repeats=repeats,extent=extent,num_epochs=num_epochs,batch_size=batchsize,learning_rate=learning_rate,
    Area=Area,version=version,special_name=special_name,model_outdir=model_outdir,databeginyear=databeginyear,
    beginyear=beginyear,endyear=endyear,augmentation=augmentation,Tranpose_Augmentation=Transpose_augmentation,Flip_Augmentation=Flip_augmentation,AddNoise_Augmentation=AddNoise_Augmentation,
    bias=bias,Normlized_PM25=Normalized_PM25,Absolute_Pm25=Absolute_PM25,Log_PM25=Log_PM25,WindSpeed_Abs=WindSpeed_Abs,GeophysicalPM25Enhenced=GeoPM25Enhenced,
    GeoPM25Enhenced_Train_infile=GeoPM25Enhenced_Training_infile,GeoPM25Enhenced_True_infile=GeoPM25Enhenced_TrueValue_infile,
    GeoPM25Enhenced_DataStartYear=GeoPM25Enforced_beginyear)
        del train_input,true_input
        gc.collect()
    CV_time_end = time.time()
    CV_time = CV_time_end - CV_time_start
    if LRP == True:
        
        train_input = np.load(train_infile)
        true_input = Learning_Object_Datasets(bias=bias,Normlized_PM25=Normalized_PM25,Absolute_PM25=Absolute_PM25,Log_PM25=Log_PM25)
        MultiyearAreaModelLRP(train_input=train_input, true_input=true_input,
                        channel_index=channel_index, kfold=kfold, repeats=repeats,
                         extent=extent,
                         Area=Area,version=version,special_name=special_name,model_outdir=model_outdir,
                         databeginyear=databeginyear,beginyear=beginyear, endyear=endyear, bias=bias, Normlized_PM25=Normalized_PM25, Absolute_Pm25=Absolute_PM25,
                         Log_PM25=Log_PM25,WindSpeed_Abs=WindSpeed_Abs,calculate=LRP_Calculation,plot=LRP_Plot)
        del train_input,true_input
        gc.collect()
    

    if OnlyCV_plot == True:
        if bias == True:
            typeName = 'PM25Bias'
        elif Normalized_PM25 == True:
            typeName = 'NormaizedPM25'
        elif Absolute_PM25 == True:
            typeName = 'AbsolutePM25'
        elif Log_PM25 == True:
            typeName = 'LogPM25'
        Area = 'EU'
        data_indic = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/GlobalTraining_MultipleModel_Spatial_withAreas_Cross_Validation_BenchMark/data_output/v' + version + '/'
        obs_pm25_outfile = data_indic + typeName+'_ObservationPM25_'+str(nchannel)+'Channel_'+Area+'_Alltime'+special_name+'.npy'
        pre_pm25_outfile = data_indic + typeName+'_PredictionPM25_'+str(nchannel)+'Channel_'+Area+'_Alltime'+special_name+'.npy'
    
        plot_from_data(infile=pre_pm25_outfile,true_infile=obs_pm25_outfile,Area=Area,version=version,special_name=special_name,nchannel=nchannel,bias=bias,
        Normlized_PM25=Normalized_PM25,Absolute_Pm25=Absolute_PM25,Log_PM25=Log_PM25)

total_time_end = time.time()

Total_time = total_time_end - total_time_start

with open(txt_outfile,'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time for CV: ',str(np.round(CV_time,4)),'\nTime for total:  ',str(np.round(Total_time,4))])