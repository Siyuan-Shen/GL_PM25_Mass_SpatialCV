import numpy as np
import torch
import time
from Estimation_pkg.data_func import get_extent_index, get_landtype
from Estimation_pkg.utils import inputfiles_table
from Training_pkg.Model_Func import predict



def map_predict(inputmap:np.array, model, train_mean:np.array,train_std:np.array, extent:list,width:int, nchannel:int,
 YYYY:str, MM:str, total_channel_names, main_stream_channel_names,side_channel_names):
    ''''''
    lat_index, lon_index = get_extent_index(extent)
    landtype = get_landtype(YYYY,extent)
    output = np.full((len(lat_index),len(lon_index)),-999.0,dtype=np.float32)
    print(YYYY,MM,' Prediction is beginning!')
    Total_start_time = time.time()
    batchsize = 5000
    for ix in range(len(lat_index)):
        land_index = np.where(landtype[ix,:] != 0)
        
        print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
        if len(land_index[0]) == 0:
            None
        else:
            temp_input = np.zeros((len(land_index[0]), nchannel, width, width), dtype=np.float32)
            GET_INPUT_TIME_START = time.time()
            for iy in range(len(land_index[0])):
                temp_input[iy,:,:,:] = inputmap[:,int(lat_index[ix] - (width - 1) / 2):int(lat_index[ix] + (width + 1) / 2), int(lon_index[land_index[0][iy]] - (width - 1) / 2):int(lon_index[land_index[0][iy]] + (width + 1) / 2)]
        
            temp_input -= train_mean
            temp_input /= train_std

            GET_INPUT_TIME_END =time.time()
            GET_INPUT_TIME = GET_INPUT_TIME_END - GET_INPUT_TIME_START
            print('Get Input Time is ', GET_INPUT_TIME, 's, the number of datasets is ', len(land_index[0]))

            GET_PREDICT_TIME_START = time.time()
            temp_output = predict(inputarray=temp_input, model=model, batchsize=batchsize,initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_channel_names)
            GET_PREDICT_TIME_END = time.time()
            GET_PREDICT_TIME = GET_PREDICT_TIME_END - GET_PREDICT_TIME_START
            print('Predict time is ', GET_PREDICT_TIME, 's, the number of datasets is ', len(land_index[0]),'batchsize: ', batchsize)

            output[ix,land_index[0]] = temp_output
    Total_end_time = time.time()
    Total_map_predict_time = Total_end_time - Total_start_time
    print(YYYY, MM, 'Prediction Ended! Time is ', Total_map_predict_time, 's', '\nShape of Map:', output.shape)

    return output

def map_final_output(output,extent,YYYY,MM, SPECIES, bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std):
    lat_index, lon_index = get_extent_index(extent)
    infiles = inputfiles_table(YYYY=YYYY,MM=MM)
    if SPECIES == 'NO3':
        SPECIES = 'NIT'
    GeoSpecies = np.load(infiles['Geo{}'.format(SPECIES)])
    
    if bias == True:
        final_data = output + GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
    elif normalize_bias == True:
        final_data = output * std + mean + GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
    elif normalize_species == True:
        final_data = output * std + mean
    elif absolute_species == True:
        final_data = output
    elif log_species == True:
        final_data = np.exp(output) - 1
    return final_data



    
