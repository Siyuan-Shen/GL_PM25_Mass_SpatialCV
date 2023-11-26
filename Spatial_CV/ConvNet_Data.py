import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .utils import pretrained_get_area_index



def get_area_index(Test_index:np.array, MONTH:int, extent:list, nsite:int):
    '''
    This function is used to return the true bias and sites PM25 values in selected area. And also return the index
    of the map (lat index and lon index) to help other functions to find the value located near the site.
    :param YEAR:
    :param MONTH:
    :param extent:
    :param nsite:
    :return:
    '''
    # *-------------------------*
    # Load files
    # *-------------------------*
    site_lat_index = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/lat_index.npy')
    site_lon_index = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/lon_index.npy')
    SATLAT = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/tSATLAT.npy')
    SATLON = np.load('/my-projects/Projects/MLCNN_PM25_2021/data/tSATLON.npy')

    lat_index = np.where((SATLAT[site_lat_index[Test_index]] >= extent[0])&(SATLAT[site_lat_index[Test_index]]<=extent[1]))
    lon_index = np.where((SATLON[site_lon_index[lat_index]] >= extent[2]) & (SATLON[site_lon_index[lat_index]] <= extent[3]))
    site_index = lat_index[0][lon_index]

    lat_offset = np.where(SATLAT >= extent[0])[0][0]
    lon_offset = np.where(SATLON >= extent[2])[0][0]
    Area_lat_index = site_lat_index[site_index] - lat_offset
    Area_lon_index = site_lon_index[site_index] - lon_offset
    return site_index, Area_lat_index, Area_lon_index



transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
])

class Dataset(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for training datasets. It is used for the global datasets, which is continuous data.
    '''
    def __init__(self, traindata, truedata):  # 'Initialization' Data Loading
        '''

        :param traindata:
            Training data.
        :param truedata:
            Ture data to learn.
        :param beginyear:
            The begin year.
        :param endyear:
            The end year.
        :param nsite:
            The number of sites. For example, for overall observation it is 10870.
        '''
        super(Dataset, self).__init__()

        self.traindatasets = torch.Tensor(traindata)  # Read training data from npy file
        self.truedatasets = torch.Tensor(truedata)
        print(self.truedatasets.shape)
        print(self.traindatasets.shape)

        self.transforms = transform  # 转为tensor形式
        self.shape = self.traindatasets.shape
    def __getitem__(self, index):  # 'Generates one sample of data'
        # Select sample
        traindata = self.traindatasets[index, :, :]
        truedata = self.truedatasets[index]
        return traindata, truedata

        # Load data and get label
    def __len__(self):  # 'Denotes the total number of samples'
        return self.traindatasets.shape[0]  # Return the total number of datasets

class Dataset_Val(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for validation datasets
    '''

    def __init__(self, traindata):  # 'Initialization' Data Loading
            super(Dataset_Val, self).__init__()

            self.traindatasets = torch.Tensor(traindata)  # Read training data from npy file


            print(self.traindatasets.shape)

            self.transforms = transform  # 转为tensor形式
            self.shape = self.traindatasets.shape

    def __getitem__(self, index):  # 'Generates one sample of data'
            # Select sample
            traindata = self.traindatasets[index, :, :]

            return traindata

            # Load data and get label

    def __len__(self):  # 'Denotes the total number of samples'
            return self.traindatasets.shape[0]  # Return the total number of datasets

class Dataset_Area(torch.utils.data.Dataset):
    def __init__(self, traindata, truedata, beginyear, endyear,):
        super(self,Dataset_Area).__init__()

def normalize_Func(inputarray:np.array):
    input_mean = np.mean(inputarray,axis=0)
    input_std  = np.std(inputarray,axis=0)
    inputarray -= input_mean
    inputarray /= input_std

    return inputarray,input_mean,input_std

def Normlize_Training_Datasets(train_input:np.array,channel_index:np.array,):

    #train_input = train_input[:, :, :, :]
    train_mean  = np.mean(train_input, axis=0)
    train_std   = np.std(train_input, axis=0)
    train_input -= train_mean
    train_input /= train_std
    train_input = np.nan_to_num(train_input, nan=0.0, posinf=0.1, neginf=-0.1)
    return train_input, train_mean, train_std

def Normlize_Testing_Datasets(true_input:np.array):
    obs_mean = np.mean(true_input)
    obs_std = np.std(true_input)
    true_input -= obs_mean
    true_input /= obs_std
    return true_input, obs_mean,obs_std


def Learning_Object_Datasets(bias:bool,Normalized_PM25Bias:bool,Normlized_PM25:bool,Absolute_PM25:bool, Log_PM25):
    input_dir = '/my-projects/Projects/MLCNN_PM25_2021/data/'
    if bias == True:
        test_infile = input_dir + 'true_data.npy'
        true_input  = np.load(test_infile)
    elif Normalized_PM25Bias == True:
        test_infile = input_dir + 'true_data.npy'
        bias_data  = np.load(test_infile)
        bias_mean  = np.mean(bias_data)
        bias_std   = np.std(bias_data)
        true_input = (bias_data - bias_mean) / bias_std
    elif Normlized_PM25 == True:
        obs_data = np.load(input_dir + 'obsPM25.npy')
        obs_mean = np.mean(obs_data)
        obs_std = np.std(obs_data)
        true_input = (obs_data - obs_mean) / obs_std
    elif Absolute_PM25 == True:
        true_input = np.load(input_dir + 'obsPM25.npy')
    elif Log_PM25 == True:
        obs_data = np.load(input_dir + 'obsPM25.npy')
        true_input = np.log(obs_data+1)
    return true_input

def Loading_Learning_Object_DatasetsforAreaNormTest(bias:bool,Normlized_PM25:bool,Absolute_PM25:bool, Log_PM25):
    input_dir = '/my-projects/Projects/MLCNN_PM25_2021/data/'
    if bias == True:
        test_infile = input_dir + 'true_data.npy'
        true_input  = np.load(test_infile)
    elif Normlized_PM25 == True:
        true_input = np.load(input_dir + 'obsPM25.npy')
        obs_mean = np.mean(true_input)
        obs_std = np.std(true_input)
        true_input = (true_input - obs_mean) / obs_std
    elif Absolute_PM25 == True:
        true_input = np.load(input_dir + 'obsPM25.npy')
    elif Log_PM25 == True:
        obs_data = np.load(input_dir + 'obsPM25.npy')
        true_input = np.log(obs_data+1)
    return true_input



def Data_Augmentation(X_train:np.array,X_test:np.array,Tranpose_Augmentation:bool,Flip_Augmentation:bool,AddNoise_Augmentation:bool):
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
    if AddNoise_Augmentation == True:
        X_addnoise_train,X_addnoise_test = AddNosie2data(X_train=X_train,X_test=X_test)
        X_train_output = np.append(X_train_output,X_addnoise_train,axis = 0)
        X_test_output = np.append(X_test_output,X_addnoise_test)
    
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

def AddNosie2data(X_train:np.array,X_test:np.array):
    X_sigma = np.std(X_train,axis=0)
    X_addnoise_train = np.zeros(X_train.shape)
    for idata in range(len(X_addnoise_train[:,0,0,0])):
        temp_coefficient = np.random.randint(-50,50,size=[30,11,11])*0.01 
        X_addnoise_train[idata,:,:,:] = X_train[idata,:,:,:] + X_sigma*temp_coefficient
    X_addnoise_test = X_test
    return X_addnoise_train,X_addnoise_test



def Get_GeophysicalPM25_Datasets(train_infile:str,true_infile:str,train_mean:np.float32,train_std:np.float32,channel_index:np.array,
        extent:np.array,GeoPM25databeginyear:int,beginyear:int,endyear:int,bias:bool, Normlized_PM25:bool, Absolute_PM25:bool,
                         Log_PM25:bool):
    train_data = np.load(train_infile)
    
    true_value = np.load(true_infile)
    if bias == True:
        true_value[:]  = 0.0
    elif Normlized_PM25 == True:
        input_dir = '/my-projects/Projects/MLCNN_PM25_2021/data/'
        obs_data = np.load(input_dir + 'obsPM25.npy')
        obs_mean = np.mean(obs_data)
        obs_std = np.std(obs_data)
        true_value -= obs_mean
        true_value /= obs_std 
    elif Absolute_PM25 == True:
        true_value = true_value
    elif Log_PM25 == True:
        true_value = np.log(true_value+1)
    nsite = len(true_value)
    site_index = np.array(range(nsite))
    area_index = pretrained_get_area_index(extent=extent,test_index=site_index)
    X_index = np.zeros((12 * (endyear - beginyear + 1) * len(area_index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        X_index[i * len(area_index):(i + 1) * len(area_index)] = ((beginyear - GeoPM25databeginyear) * 12 + i) * nsite + area_index
    
    train_input = train_data[X_index,:,:,:]
    train_input = train_input[:,channel_index,:,:]
    train_input -= train_mean[channel_index,:,:]
    train_input /= train_std[channel_index,:,:]
    true_input  = true_value[X_index]
    return train_input,true_input