import csv
import numpy as np
from Spatial_CV.data_func import *



def save_loss_accuracy(model_outdir, TrainingOrTesting, loss, accuracy, typeName, epoch, nchannel, special_name, width, height):

    outdir = model_outdir + '/Results/results-Trained_Models/'
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    loss_outfile = outdir + 'SpatialCV_{}_loss_{}_{}Epoch_{}x{}_{}Channel{}.npy'.format(TrainingOrTesting, typeName, epoch, width, height, nchannel,special_name)
    accuracy_outfile = outdir + 'SpatialCV_{}_accuracy_{}_{}Epoch_{}x{}_{}Channel{}.npy'.format(TrainingOrTesting, typeName, epoch, width, height, nchannel,special_name)
    np.save(loss_outfile, loss)
    np.save(accuracy_outfile, accuracy)
    return

def output_text(outfile:str,status:str,Areas:list,Area_beginyears:dict,endyear:int,
                test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, slope_CV_R2, PWAModel, PWAMonitors):
    
    MONTH = ['Annual','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_CV_R2_Alltime, slope_CV_R2_Alltime, PWAModel_Alltime, PWAMonitors_Alltime = calculate_Alltime_Statistics_results(Areas,Area_beginyears,endyear,test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, slope_CV_R2, PWAModel, PWAMonitors)

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        for iarea in Areas:
            writer.writerow(['Area: {} ; Time Period: {} - {}'.format(iarea, Area_beginyears[iarea], endyear)])
        
            for imonth in MONTH:
                writer.writerow(['-------------------------- {} ------------------------'.format(imonth), 
                            '\n Test R2 - Avg: ', str(np.round(test_CV_R2_Alltime[iarea]['Alltime'][imonth][0], 4)), 'Min: ',
                             str(np.round(test_CV_R2_Alltime[iarea]['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(test_CV_R2_Alltime[iarea]['Alltime'][imonth][2],4)),

                             '\n Slope - Avg: ', str(np.round(slope_CV_R2_Alltime[iarea]['Alltime'][imonth][0], 4)), 'Min: ',
                             str(np.round(slope_CV_R2_Alltime[iarea]['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(slope_CV_R2_Alltime[iarea]['Alltime'][imonth][2],4)),

                             '\n RMSE -  Avg: ', str(np.round(RMSE_CV_R2_Alltime[iarea]['Alltime'][imonth][0], 4)), 'Min: ',
                             str(np.round(RMSE_CV_R2_Alltime[iarea]['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(RMSE_CV_R2_Alltime[iarea]['Alltime'][imonth][2],4)),

                             '\n Training R2 - Avg: ',str(np.round(train_CV_R2_Alltime[iarea]['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(train_CV_R2_Alltime[iarea]['Alltime'][imonth][1], 4)), 'Max: ',
                             str(np.round(train_CV_R2_Alltime[iarea]['Alltime'][imonth][2],4)),

                             '\n Geophysical R2 - Avg: ',str(np.round(geo_CV_R2_Alltime[iarea]['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(geo_CV_R2_Alltime[iarea]['Alltime'][imonth][1], 4)), 'Max: ',
                             str(np.round(geo_CV_R2_Alltime[iarea]['Alltime'][imonth][2],4)), 
                             
                             '\n PWA Model - Avg: ',str(np.round(PWAModel_Alltime[iarea]['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAModel_Alltime[iarea]['Alltime'][imonth][1], 4)), 'Max: ',
                             str(np.round(PWAModel_Alltime[iarea]['Alltime'][imonth][2],4)), 

                             '\n PWA co-monitors - Avg: ',str(np.round(PWAMonitors_Alltime[iarea]['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAMonitors_Alltime[iarea]['Alltime'][imonth][1], 4)), 'Max: ',
                             str(np.round(PWAMonitors_Alltime[iarea]['Alltime'][imonth][2],4)), 
                             ])
                

    return