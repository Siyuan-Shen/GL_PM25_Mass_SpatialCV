
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader
from Spatial_CV.Statistic_Func import linear_regression
from Spatial_CV.ConvNet_Data import Dataset,Dataset_Val
import torch.nn.functional as F


nsite = 10870


class GeophysicalTwoPenaltiesLoss(nn.Module):
    def __init__(self,lambda1:float,lambda2:float,alpha:float,beta:float,PM25Bias:bool,size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(GeophysicalTwoPenaltiesLoss, self).__init__()
        ### lambda1 and lambda2 must be negative
        if lambda1 < 0:
            self.lambda1 = 1.0
        else:
            self.lambda1 = lambda1
        
        if lambda2 < 0:
            self.lambda2 = 1.0
        else:
            self.lambda2 = lambda2
        
        self.PM25Bias = PM25Bias
        self.alpha = np.abs(alpha)
        self.beta  = np.abs(beta)
        #self.GeoPM25_mean = GeoPM25_mean
        #self.GeoPM25_std  = GeoPM25_std
        ##### Typical Parameters Settings
        self.reduction = reduction
    def forward(self, input: torch.Tensor, target: torch.Tensor, geophysical:torch.Tensor,GeoPM25_mean:np.float32,GeoPM25_std:np.float32) -> torch.Tensor:
        #### input is normalized
        if self.PM25Bias == True:
            print('Here is PM2.5 Bias')
            input = input
        else:
            mean = 22.37
            std = 23.528
            input = input * std + mean
        MSE_LOSS =  F.mse_loss(input, target, reduction=self.reduction)
        geophysical = geophysical * GeoPM25_std + GeoPM25_mean
        #Penalty1 =  F.mse_loss(input,0.1*geophysical,reduction=self.reduction)

        Penalty1 = torch.sum(torch.relu(-input - self.alpha*geophysical))
        Penalty2 = torch.sum(torch.relu(input - self.beta*geophysical))
        print('MSELoss: ', MSE_LOSS,'\nPenalty1: ', Penalty1, '\nPenalty2: ', Penalty2)
        Loss = MSE_LOSS + self.lambda1 * Penalty1 + self.lambda2 * Penalty2
        return Loss

class MyLoss(nn.Module):
    def __init__(self,lambda1:float,lambda2:float,minbar:float,maxbar:float,size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MyLoss, self).__init__()
        ### lambda1 and lambda2 must be negative
        if lambda1 > 0:
            self.lambda1 = -0.01
        else:
            self.lambda1 = lambda1
        if lambda2 > 0:
            self.lambda2 = -0.005
        else:
            self.lambda2 = lambda2
        ##### Typical Parameters Settings
        self.reduction = reduction
        self.minbar = minbar
        self.maxbar = maxbar
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        MSE_LOSS =  F.mse_loss(input, target, reduction=self.reduction)
        Penalty1 =  torch.sum(torch.relu(self.minbar - input))
        Penalty2 =  torch.sum(torch.relu(input - self.maxbar))
        print('MSELoss: ', MSE_LOSS,'\nPenalty1: ', Penalty1,'\nPenalty2:', Penalty2)
        Loss = MSE_LOSS + self.lambda1 *Penalty1 + self.lambda2 *Penalty2
        return Loss

class ElevationRewardsLoss(nn.Module):
    def __init__(self,lambda1:float,lambda2:float,minbar:float,maxbar:float,size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ElevationRewardsLoss, self).__init__()
        ### lambda1 and lambda2 must be negative
        if lambda1 > 0:
            self.lambda1 = -0.01
        else:
            self.lambda1 = lambda1
        if lambda2 > 0:
            self.lambda2 = -0.005
        else:
            self.lambda2 = lambda2
        ##### Typical Parameters Settings
        self.reduction = reduction
        self.minbar = minbar
        self.maxbar = maxbar
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        MSE_LOSS =  F.mse_loss(input, target, reduction=self.reduction)
        Penalty1 =  torch.mean(torch.relu(self.minbar - input))
        Penalty2 =  torch.mean(torch.relu(input - self.maxbar))
        print('MSELoss: ', MSE_LOSS,'\nPenalty1: ', Penalty1,'\nPenalty2:', Penalty2)
        Loss = MSE_LOSS + self.lambda1 *Penalty1 + self.lambda2 *Penalty2
        return Loss
class SigmoidMSELoss(nn.Module):
    def __init__(self,alpha:float,beta:float,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SigmoidMSELoss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input:torch.Tensor,target:torch.Tensor)->torch.Tensor:
        sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
        MSE_Loss = F.mse_loss(sigmoid_coefficient*input,sigmoid_coefficient*target)
        print('MSELoss: ', MSE_Loss)
        return MSE_Loss
class SigmoidMSELossWithGeoSumPenalties_withAbsoluteLimitation(nn.Module):
    def __init__(self,alpha:float,beta:float,lambda1:float,lambda3:float,gamma:float,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SigmoidMSELossWithGeoSumPenalties_withAbsoluteLimitation,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        self.reduction = reduction
    def forward(self, input:torch.Tensor,target:torch.Tensor,geophysical:torch.Tensor,GeoPM25_mean:np.float32,GeoPM25_std:np.float32)->torch.Tensor:
        sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
        MSE_Loss = F.mse_loss(sigmoid_coefficient*input,sigmoid_coefficient*target)

        geophysical = geophysical * GeoPM25_std + GeoPM25_mean
        Penalty1 = torch.sum(torch.relu(-input -  geophysical))
        Penalty2 = torch.sum(torch.relu(input - self.gamma * geophysical))
        Penalty3 = torch.sum(torch.relu(input - 80.0))

        print('MSELoss: ', MSE_Loss, '\nPenalty2: ', Penalty2, '\nPenalty3: ', Penalty3)
        Loss = MSE_Loss + self.lambda1 * Penalty2 + self.lambda3* Penalty3#10*Penalty1
        print('MSELoss: ', MSE_Loss)

        return Loss

class SigmoidMSELossWithGeoSumPenalties(nn.Module):
    def __init__(self,alpha:float,beta:float,lambda1:float,gamma:float,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SigmoidMSELossWithGeoSumPenalties,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.reduction = reduction
    def forward(self, input:torch.Tensor,target:torch.Tensor,geophysical:torch.Tensor,GeoPM25_mean:np.float32,GeoPM25_std:np.float32)->torch.Tensor:
        sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
        geophysical = geophysical * GeoPM25_std + GeoPM25_mean
        #sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target+geophysical)))+1)
        
        MSE_Loss = F.mse_loss(sigmoid_coefficient*input,sigmoid_coefficient*target)

        
        Penalty1 = torch.sum(torch.relu(-input - geophysical))
        Penalty2 = torch.sum(torch.relu(input - self.gamma * geophysical))

        print('MSELoss: ', MSE_Loss, '\nPenalty2: ', Penalty2)
        Loss = MSE_Loss + self.lambda1 * Penalty2 + 10 * Penalty1
        print('MSELoss: ', MSE_Loss)
        return Loss

class SigmoidMSELoss_WithGeoSitesNumberSumPenalties(nn.Module):
    def __init__(self,alpha:float,beta:float,lambda1:float,gamma:float,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SigmoidMSELoss_WithGeoSitesNumberSumPenalties,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.reduction = reduction
    def forward(self, input:torch.Tensor,target:torch.Tensor,geophysical:torch.Tensor,GeoPM25_mean:np.float32,GeoPM25_std:np.float32,
                SitesNumber:np.float32,SitesNumber_mean:np.float32,SitesNumber_std:np.float32)->torch.Tensor:
        sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
        MSE_Loss = F.mse_loss(sigmoid_coefficient*input,sigmoid_coefficient*target)

        geophysical = geophysical * GeoPM25_std + GeoPM25_mean
        SitesNumber = SitesNumber * SitesNumber_std + SitesNumber_mean
        
        Penalty1 = torch.sum((1/SitesNumber)*torch.relu(-input - geophysical))
        Penalty2 = torch.sum((1/SitesNumber)*torch.relu(input - self.gamma * geophysical))

        print('MSELoss: ', MSE_Loss, '\nPenalty2: ', Penalty2)
        Loss = MSE_Loss + self.lambda1 * Penalty2 + 10*Penalty1
        print('Total Loss: ', Loss)
        return Loss


class SigmoidMSELoss_WithGeoSitesNumberLogPenalties(nn.Module):
    def __init__(self,alpha:float,beta:float,lambda1:float,gamma:float,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SigmoidMSELoss_WithGeoSitesNumberLogPenalties,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.reduction = reduction
    def forward(self, input:torch.Tensor,target:torch.Tensor,geophysical:torch.Tensor,GeoPM25_mean:np.float32,GeoPM25_std:np.float32,
                SitesNumber:np.float32,SitesNumber_mean:np.float32,SitesNumber_std:np.float32)->torch.Tensor:
        sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
        MSE_Loss = F.mse_loss(sigmoid_coefficient*input,sigmoid_coefficient*target)

        geophysical = geophysical * GeoPM25_std + GeoPM25_mean
        SitesNumber = SitesNumber * SitesNumber_std + SitesNumber_mean
        
        #Penalty1 = torch.sum((1/SitesNumber)*torch.relu(-input - geophysical))
        Penalty2 = torch.sum((1/SitesNumber)*torch.relu(torch.abs(torch.log(1+input/geophysical)) - np.log(1.0+self.gamma)))

        print('MSELoss: ', MSE_Loss, '\nPenalty2: ', Penalty2)
        Loss = MSE_Loss + self.lambda1 * Penalty2
        print('Total Loss: ', Loss)

        return Loss
class SigmoidMSELoss_WithSqrtGeoSitesNumberLogPenalties(nn.Module):
    def __init__(self,alpha:float,beta:float,lambda1:float,gamma:float,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SigmoidMSELoss_WithSqrtGeoSitesNumberLogPenalties,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.reduction = reduction
    def forward(self, input:torch.Tensor,target:torch.Tensor,geophysical:torch.Tensor,GeoPM25_mean:np.float32,GeoPM25_std:np.float32,
                SitesNumber:np.float32,SitesNumber_mean:np.float32,SitesNumber_std:np.float32)->torch.Tensor:
        sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
        MSE_Loss = F.mse_loss(sigmoid_coefficient*input,sigmoid_coefficient*target)

        geophysical = geophysical * GeoPM25_std + GeoPM25_mean
        SitesNumber = SitesNumber * SitesNumber_std + SitesNumber_mean
        
        #Penalty1 = torch.sum((1/SitesNumber)*torch.relu(-input - geophysical))
        Penalty2 = torch.sum((torch.sqrt(1/SitesNumber))*torch.relu(torch.abs(torch.log(1+input/geophysical)) - np.log(1.0+self.gamma)))

        print('MSELoss: ', MSE_Loss, '\nPenalty2: ', Penalty2)
        Loss = MSE_Loss + self.lambda1 * Penalty2
        print('Total Loss: ', Loss)

        return Loss
class SigmoidMSELoss_WithExpGeoSitesNumberLogPenalties(nn.Module):
    def __init__(self,alpha:float,beta:float,lambda1:float,lambda2:float,gamma:float,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SigmoidMSELoss_WithExpGeoSitesNumberLogPenalties,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.reduction = reduction
    def forward(self, input:torch.Tensor,target:torch.Tensor,geophysical:torch.Tensor,GeoPM25_mean:np.float32,GeoPM25_std:np.float32,
                SitesNumber:np.float32,SitesNumber_mean:np.float32,SitesNumber_std:np.float32)->torch.Tensor:
        sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
        MSE_Loss = F.mse_loss(sigmoid_coefficient*input,sigmoid_coefficient*target)

        geophysical = geophysical * GeoPM25_std + GeoPM25_mean
        SitesNumber = SitesNumber * SitesNumber_std + SitesNumber_mean
        
        #Penalty1 = torch.sum((1/SitesNumber)*torch.relu(-input - geophysical))
        Penalty2 = torch.sum((self.lambda1*torch.exp(-self.lambda2*torch.pow(SitesNumber,4)))*torch.relu(torch.abs(torch.log(1+input/geophysical)) - np.log(1.0+self.gamma)))

        print('MSELoss: ', MSE_Loss, '\nSymmetric Penalty2: ', Penalty2)
        Loss = MSE_Loss + self.lambda1 * Penalty2
        print('Total Loss: ', Loss)

        return Loss
    
def train(model, X_train, y_train, BATCH_SIZE, learning_rate, TOTAL_EPOCHS, GeoPM25_mean, GeoPM25_std,
          SitesNumber_mean, SitesNumber_std):

    train_loader = DataLoader(Dataset(X_train, y_train), BATCH_SIZE, shuffle=True)


    print('*' * 25, type(train_loader), '*' * 25)
    #criterion = nn.SmoothL1Loss()
    #criterion = nn.MSELoss()

    #alpha = 0.005
    #beta = 8.0
    #criterion = SigmoidMSELoss(alpha=alpha,beta=beta)
    #print('Sigmoid MSELoss alpha: ',alpha, ' beta: ',beta)

    #alpha = 0.005
    #beta = 8.0
    #gamma = 3.0
    #lambda1 = 0.1
    #criterion = SigmoidMSELossWithGeoSumPenalties(alpha=alpha,beta=beta,lambda1=lambda1,gamma=gamma)


    
    alpha = 0.0005
    beta = 5.0
    gamma = 3.0
    lambda1 = 0.2
    lambda2 = 5e-7
    #criterion = SigmoidMSELossWithGeoSumPenalties_withAbsoluteLimitation(alpha=alpha,beta=beta,gamma=gamma,lambda1=lambda1,lambda3=lambda3)
    criterion = SigmoidMSELossWithGeoSumPenalties(alpha=alpha,beta=beta,lambda1=lambda1,gamma=gamma)
    #criterion =SigmoidMSELoss_WithGeoSitesNumberSumPenalties(alpha=alpha,beta=beta,lambda1=lambda1,gamma=gamma)
    #criterion = SigmoidMSELoss_WithGeoSitesNumberLogPenalties(alpha=alpha,beta=beta,lambda1=lambda1,gamma=gamma）
    #criterion = SigmoidMSELoss_WithSqrtGeoSitesNumberLogPenalties(alpha=alpha,beta=beta, lambda1=lambda1, gamma=gamma)
    #criterion  = SigmoidMSELoss_WithExpGeoSitesNumberLogPenalties(alpha=alpha,beta=beta,gamma=gamma,lambda1=lambda1,lambda2=lambda2)


    #lambda1 = 0.5
    #lambda2 = 0.5
    #alpha  = 1.0
    #beta   = 1.0
    #criterion = GeophysicalTwoPenaltiesLoss(lambda1=lambda1,lambda2=lambda2,alpha=alpha,beta=beta,PM25Bias=True)
    #print('Geophysical Loss Parameters: ','\nlambda1: ',lambda1,' lambda2: ', lambda2, ' alpha:',alpha,' beta: ',beta)



    #lambda1,lambda2,minbar,maxbar  = -0.8, -0.5, 8, 80
    #print('My Loss Parameters: ','\nlambda1: ',lambda1,'\nlambda2: ', lambda2, '\nmin bar: ',minbar,'\nmaxbar: ',maxbar)
    #criterion = MyLoss(lambda1=lambda1,lambda2=lambda2,minbar=minbar,maxbar=maxbar)
    losses = []
    train_acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    ## scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3,threshold=0.005)
    model.train()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(TOTAL_EPOCHS):
        #learning_rate = learning_rate/(1+0.95*epoch)
        correct = 0
        counts = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = torch.squeeze(labels.type(torch.FloatTensor))
            labels = labels.to(device)
            optimizer.zero_grad()  # Set grads to zero
            outputs = model(images) #dimension: Nx1
            outputs = torch.squeeze(outputs)
            #print(outputs)
            # print('output.shape,labels.shape :', outputs, labels)
            ## Calculate Loss Func
            loss = criterion(outputs, labels, images[:,16,5,5],GeoPM25_mean,GeoPM25_std)#,images[:,-1,5,5],SitesNumber_mean,SitesNumber_std)
            loss.backward()  ## backward
            optimizer.step()  ## refresh training parameters
            losses.append(loss.item())
            # Calculate R2
            y_hat = model(images).cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
            #torch.cuda.empty_cache()
            print('Epoch: ', epoch, ' i th: ', i)
            #print('y_hat:', y_hat)
            R2 = linear_regression(y_hat,y_true)
            R2 = np.round(R2, 4)
            #pred = y_hat.max(1, keepdim=True)[1] # 得到最大值及索引，a.max[0]为最大值，a.max[1]为最大值的索引
            correct += R2
            counts  += 1
            if (i + 1) % 10 == 0:
                # 每10个batches打印一次loss
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    loss.item()))
        accuracy = correct / counts

        print('Epoch: ',epoch, ', Loss: ', loss.item(),', Training set accuracy:',accuracy)
        train_acc.append(accuracy)
        print('Epoch: ',epoch,'\nLearning Rate:',optimizer.param_groups[0]['lr'])
        scheduler.step()
       
        # Each epoch calculate test data accuracy

    return losses,  train_acc
def predict(inputarray, model, Width, batchsize):
    #output = np.zeros((), dtype = float)
    model.eval()
    final_output = []
    final_output = np.array(final_output)
    predictinput = DataLoader(Dataset_Val(inputarray), batch_size= batchsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, image in enumerate(predictinput):
            image = image.to(device)
            output = model(image).cpu().detach().numpy()
            final_output = np.append(final_output,output)

    return final_output
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def weight_init_normal(m):  #初始化权重 normal
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()

def initialize_weights_kaiming(m): #kaiming_uniform
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='tanh')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)


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


