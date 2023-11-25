import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
import math
import os
from Spatial_CV.Statistic_Func import linear_regression,linear_slope,regress2
from Spatial_CV.utils import *
import seaborn as sns
####################################################################
##                       Plotting Sets Variables                  ##
####################################################################

'''
nrows = 2
ncols = 2
proj = ccrs.PlateCarree()
aspect = (179+179)/(60+70)
height = 5.0
width = aspect * height
vpad = 0.03 * height
hpad = 0.02 * width
hlabel = 0.12 * height*2
vlabel = 0.1 * height*2
hmargin = 0.03 * width
vmargin = 0.03 * height*2
cbar_height = 0.48 * height
cbar_width = 0.015 * width
cbar_height_2 = 0.9 * (height*2 - vlabel)
cbar_width_2 = 0.08 * (width + height*2)
'''

nrows = 2
ncols = 2
proj = ccrs.PlateCarree()
aspect = (179)/(60+70)
height = 5.0
width = aspect * height
vpad = 0.03 * height
hpad = 0.02 * width
hlabel = 0.12 * height*2
vlabel = 0.1 * height*2
hmargin = 0.03 * width
vmargin = 0.03 * height*2
cbar_height = 0.48 * height
cbar_width = 0.015 * width
cbar_height_2 = 0.9 * (height*2 - vlabel)
cbar_width_2 = 0.08 * (width + height*2)

figwidth = width + height + hmargin*2 + cbar_width_2
figheight = height*2 + vmargin*2

def regression_plot_ReducedAxisReduced(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    version:str, channel:int, special_name:str, area_name:str,beginyear:str,endyear:str, extentlim:int,
                    bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,
                    Log_PM25:bool):
    fig_output_dir = Scatter_plots_outdir + '{}/figures/scatter_figures/'.format(version)
    if not os.path.isdir(fig_output_dir):
        os.makedirs(fig_output_dir)
    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    
    fig_outfile =  fig_output_dir + typeName+'_PM25_RMARegressionPlot_'+str(channel)+'Channel_'+area_name+'_'+beginyear+endyear+special_name+'.png'
    data_outdic = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/GlobalTraining_MultipleModel_Spatial_withAreas_Cross_Validation_BenchMark/data_output/v' + version + '/'
    if not os.path.isdir(data_outdic):
        os.makedirs(data_outdic)
    obs_pm25_outfile = data_outdic + typeName+'_ObservationPM25_'+str(channel)+'Channel_'+area_name+'_'+beginyear+endyear+special_name+'.npy'
    pre_pm25_outfile = data_outdic + typeName+'_PredictionPM25_'+str(channel)+'Channel_'+area_name+'_'+beginyear+endyear+special_name+'.npy'
    np.save(obs_pm25_outfile,plot_obs_pm25)
    np.save(pre_pm25_outfile,plot_pre_pm25)

    H, xedges, yedges = np.histogram2d(plot_obs_pm25, plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(plot_obs_pm25, plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(plot_obs_pm25, plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([(4 * hmargin + hlabel) / figwidth, (vmargin + vlabel) / figheight,
                   (height * 2 - hlabel) / figwidth,
                   (height * 2 - vlabel) / figheight])  # [left, bottom, width, height]
    cbar_ax = plt.axes([(height * 2 + 0.8 * cbar_width_2) / figwidth,
                        (vmargin + vlabel + 0.05 * (height * 2 - vlabel)) / figheight,
                        0.2 * cbar_width_2 / figwidth, cbar_height_2 / figheight])
    regression_Dic = regress2(_x=plot_obs_pm25,_y=plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis')
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    r = regression_Dic['r']
    print('RMA_divide intercept=0')
    b0 = 0#round(b0, 2)
    b1 = round(b1/r, 2)

    extentlim = extentlim
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(plot_obs_pm25, plot_pre_pm25,
                   cmap='hot_r', norm=colors.LogNorm(vmin=1, vmax=100), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='blue', linestyle='-')
    ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Observation $PM_{2.5}$ concentration, unit = $\mu g/m^3$', fontsize=24)
    ax.set_ylabel('Estimated $PM_{2.5}$ concentration, unit = $\mu g/m^3$', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $' + str(R2), style='italic', fontsize=32)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'$\mu g/m^3$', style='italic', fontsize=32)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0) + ' + ' + str(b1) + 'x', style='italic',
            fontsize=32)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0) + 'x', style='italic',
            fontsize=32)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0) + ' - ' + str(abs(b1)) + 'x', style='italic',
            fontsize=32)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(plot_pre_pm25)), style='italic',
            fontsize=32)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$',], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)

    fig.savefig(fig_outfile)

    
    return
def regression_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    version:str, channel:int, special_name:str, area_name:str,beginyear:str,endyear:str, extentlim:int,
                    bias:bool, Normlized_PM25:bool, Absolute_Pm25:bool,
                    Log_PM25:bool) -> None:
    fig_output_dir = Scatter_plots_outdir + '{}/Figures/scatter-figures/'.format(version)

    if not os.path.isdir(fig_output_dir):
        os.makedirs(fig_output_dir)
    if bias == True:
        typeName = 'PM25Bias'
    elif Normlized_PM25 == True:
        typeName = 'NormaizedPM25'
    elif Absolute_Pm25 == True:
        typeName = 'AbsolutePM25'
    elif Log_PM25 == True:
        typeName = 'LogPM25'
    fig_outfile =  fig_output_dir + '{}_PM25_RegressionPlot_{}Channel_{}_{}-{}{}.png'.format(typeName,channel,area_name,beginyear,endyear,special_name)
    
    data_outdic = model_outdir + '{}/data_recording/'.format(version)
    if not os.path.isdir(data_outdic):
        os.makedirs(data_outdic)
    obs_pm25_outfile = data_outdic + typeName+'_ObservationPM25_'+str(channel)+'Channel_'+area_name+'_'+beginyear+endyear+special_name+'.npy'
    pre_pm25_outfile = data_outdic + typeName+'_PredictionPM25_'+str(channel)+'Channel_'+area_name+'_'+beginyear+endyear+special_name+'.npy'
    np.save(obs_pm25_outfile,plot_obs_pm25)
    np.save(pre_pm25_outfile,plot_pre_pm25)

    H, xedges, yedges = np.histogram2d(plot_obs_pm25, plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(plot_obs_pm25, plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(plot_obs_pm25, plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([(4 * hmargin + hlabel) / figwidth, (vmargin + vlabel) / figheight,
                   (height * 2 - hlabel) / figwidth,
                   (height * 2 - vlabel) / figheight])  # [left, bottom, width, height]
    cbar_ax = plt.axes([(height * 2 + 0.8 * cbar_width_2) / figwidth,
                        (vmargin + vlabel + 0.05 * (height * 2 - vlabel)) / figheight,
                        0.2 * cbar_width_2 / figwidth, cbar_height_2 / figheight])
    regression_Dic = regress2(_x=plot_obs_pm25,_y=plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis',
    )
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    b0 = round(b0, 2)
    b1 = round(b1, 2)

    extentlim = extentlim
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(plot_obs_pm25, plot_pre_pm25,
                   cmap='hot_r', norm=colors.LogNorm(vmin=1, vmax=100), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='blue', linestyle='-')
    #ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Observed $PM_{2.5}$ concentration ($\mu g/m^3$)', fontsize=32)
    ax.set_ylabel('Estimated $PM_{2.5}$ concentration ($\mu g/m^3$)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $0.70', style='italic', fontsize=32)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'$\mu g/m^3$', style='italic', fontsize=32)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = {}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0), style='italic',
            fontsize=32)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y=-{}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(plot_pre_pm25)), style='italic',
            fontsize=32)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$',], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)

    fig.savefig(fig_outfile)
    plt.close()

def regression_plot_area_test_average(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    version:str, channel:int, special_name:str, area_name:str, extentlim:int, time:str, fold:int):
    fig_output_dir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/GlobalTraining_MultipleModel_Spatial_withAreas_Cross_Validation_BenchMark/figures/Area_Tests/scatter_figures/' + version
    fig_outfile =  fig_output_dir +'/+ typeName+' + 'v'+\
        version +'_' + 'Spatial_CrossValidation_predict_FinalPM25Compare_fold'+str(fold)+'_'+time+'Average_'+str(channel)+'Channel_'+area_name+special_name+'.png'

    Month_Name = {'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', '06': 'June',
                  '07': 'July', '08': 'August', '09': 'September', '10': 'October', '11': 'November', '12': 'December', 'Annual': 'Annual' }
    extentlim_dic = {'NA':30,'SA':30,'EU':40,'AF':60,'AS':120,'AU':30}
    H, xedges, yedges = np.histogram2d(plot_obs_pm25, plot_pre_pm25, bins=1)
    fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(plot_obs_pm25, plot_pre_pm25))
    RMSE = round(RMSE, 4)

    R2 = linear_regression(plot_obs_pm25, plot_pre_pm25)
    R2 = round(R2, 4)

    ax = plt.axes([(4 * hmargin + hlabel) / figwidth, (vmargin + vlabel) / figheight,
                   (height * 2 - hlabel) / figwidth,
                   (height * 2 - vlabel) / figheight])  # [left, bottom, width, height]
    cbar_ax = plt.axes([(height * 2 + 0.8 * cbar_width_2) / figwidth,
                        (vmargin + vlabel + 0.05 * (height * 2 - vlabel)) / figheight,
                        0.2 * cbar_width_2 / figwidth, cbar_height_2 / figheight])

    b0, b1 = linear_slope(plot_obs_pm25,
                          plot_pre_pm25)
    b0 = round(b0, 3)
    b1 = round(b1, 3)

    extentlim = extentlim_dic[area_name]
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(plot_obs_pm25, plot_pre_pm25,
                   cmap='hot_r', norm=colors.LogNorm(vmin=1, vmax=100), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='blue', linestyle='-')
    ax.set_xlabel('Observation $PM_{2.5}$ concentration, unit = $\mu g/m^3$', fontsize=24)
    ax.set_ylabel('Estimated $PM_{2.5}$ concentration, unit = $\mu g/m^3$', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $' + str(R2), style='italic', fontsize=24)
    ax.text(0, extentlim - (0.05 + 0.044) * extentlim, '$RMSE = $' + str(RMSE), style='italic', fontsize=24)
    ax.text(0, extentlim - (0.05 + 0.044 * 2) * extentlim, 'y = ' + str(b0) + ' + ' + str(b1) + 'x', style='italic',
            fontsize=24)
    ax.text(0, extentlim - (0.05 + 0.044 * 3) * extentlim, 'N = ' + str(len(plot_pre_pm25)), style='italic',
            fontsize=24)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', '100', ], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)
    plt.title("Observation and modeled $PM_{2.5}$ scatters in " + area_name + ' Area at' + Month_Name[time])
    fig.savefig(fig_outfile)
    plt.close()

def bias_regression_plot(plot_obs_bias:np.array,plot_pre_bias:np.array,
                    version:str, channel:int, special_name:str,area_name:str):
    fig_output_dir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/Spatial_withAreas_Cross_Validation_BenchMark/figures/scatter_figures/'
    fig_outfile =  fig_output_dir +'histgram2d_' + 'v'+\
        version +'_' + 'Spatial_CrossValidation_predict_BiasCompare_'+str(channel)+'Channel_'+area_name+special_name+'.png'

    H, xedges, yedges = np.histogram2d(plot_obs_bias, plot_pre_bias, bins=100)
    fig = plt.figure(figsize=(figwidth, figheight))
    extent = [-max(xedges), max(xedges), -max(xedges), max(xedges)]
    RMSE = np.sqrt(mean_squared_error(plot_obs_bias, plot_pre_bias))
    RMSE = round(RMSE, 4)

    R2 = linear_regression(plot_obs_bias, plot_pre_bias)
    R2 = round(R2, 4)

    ax = plt.axes([(4 * hmargin + hlabel) / figwidth, (vmargin + vlabel) / figheight,
                   (height * 2 - hlabel) / figwidth,
                   (height * 2 - vlabel) / figheight])  # [left, bottom, width, height]
    cbar_ax = plt.axes([(height * 2 + 0.8 * cbar_width_2) / figwidth,
                        (vmargin + vlabel + 0.05 * (height * 2 - vlabel)) / figheight,
                        0.2 * cbar_width_2 / figwidth, cbar_height_2 / figheight])

    b0, b1 = linear_slope(plot_obs_bias,
                          plot_pre_bias)
    b0 = round(b0, 3)
    b1 = round(b1, 3)

    extentlim = 60
    minus_extentlim = -60
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(plot_obs_bias, plot_pre_bias,
                   cmap='hot_r', norm=colors.LogNorm(vmin=1, vmax=1e4), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([minus_extentlim, extentlim], [minus_extentlim, extentlim], color='black', linestyle='--')
    ax.plot([minus_extentlim, extentlim], [b0 + b1 * minus_extentlim, b0 + b1 * extentlim], color='blue', linestyle='-')
    ax.set_xlabel('Observation $PM_{2.5}$ concentration, unit = $\mu g/m^3$', fontsize=24)
    ax.set_ylabel('Estimated $PM_{2.5}$ concentration, unit = $\mu g/m^3$', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.text(minus_extentlim, extentlim - minus_extentlim - 0.1 * extentlim, '$R^2 = $' + str(R2), style='italic', fontsize=24)
    ax.text(minus_extentlim, extentlim - minus_extentlim - (0.1 + 0.088) * extentlim, '$RMSE = $' + str(RMSE), style='italic', fontsize=24)
    ax.text(0, extentlim - (0.05 + 0.044 * 2) * extentlim, 'y = ' + str(b0) + ' + ' + str(b1) + 'x', style='italic',
            fontsize=24)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100, 1e3, 1e4, 1e5, 1e6])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)

    fig.savefig(fig_outfile)
    plt.close()


def PM25_histgram_distribution_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    version:str, channel:int, special_name:str, area_name:str,bins:int,range:tuple,):
    fig_output_dir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/Spatial_withAreas_Cross_Validation_BenchMark/figures/histgram_figures/'
    fig_outfile = fig_output_dir + 'histgram2d_' + 'v' + \
                  version + '_' + 'Spatial_CrossValidation_predict_Distribution_' + str(
        channel) + 'Channel_' + area_name + special_name + '.png'

    # 添加x轴和y轴标签
    fig, ax = plt.subplots()
    plt.ylabel("Numbers(#)")
    plt.xlabel("PM2.5 Concentration (${\mu}g/m^3$)")
    plt.title("Observation and modeled $PM_{2.5}$ Distribution in "+area_name+' Area')

    ax.hist(plot_obs_pm25, bins=bins, range=range, density=False, histtype='stepfilled', color='blue',
             label='Observation', alpha=0.3)
    ax.hist(plot_pre_pm25, bins=bins, range=range, density=False, histtype='stepfilled', color='orange',
             label='ML produced', alpha=0.3)
    legend = ax.legend(loc='upper right', shadow=False, fontsize='x-large')
    plt.savefig(fig_outfile, dpi=2000)


def PM25_histgram_distribution_area_tests_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    version:str, channel:int, special_name:str, area_name:str,bins:int,range:tuple,time:str, fold:int):
    fig_output_dir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/Spatial_withAreas_Cross_Validation_BenchMark/figures/Area_Tests/histgram_figures/'+version
    fig_outfile = fig_output_dir + '/histgram2d_' + 'v' + \
                  version + '_' + 'Spatial_CrossValidation_predict_Distribution_fold'+str(fold)+'_'+ time+'_' + str(
        channel) + 'Channel_' + area_name + special_name + '.png'
    Month_Name = {'01': 'January','02':'February','03':'March','04':'April','05':'May','06':'June',
                  '07':'July','08':'August','09':'September','10':'October','11':'November','12':'December', 'Annual': 'Annual' }
    # 添加x轴和y轴标签
    fig, ax = plt.subplots()
    plt.ylabel("Numbers(#)")
    plt.xlabel("PM2.5 Concentration (${\mu}g/m^3$)")
    plt.title("Observation and modeled $PM_{2.5}$ Distribution in "+area_name+' Area at' + Month_Name[time])

    ax.hist(plot_obs_pm25, bins=bins, range=range, density=False, histtype='stepfilled', color='blue',
             label='Observation', alpha=0.3)
    ax.hist(plot_pre_pm25, bins=bins, range=range, density=False, histtype='stepfilled', color='orange',
             label='ML produced', alpha=0.3)
    legend = ax.legend(loc='upper right', shadow=False, fontsize='x-large')
    plt.savefig(fig_outfile, dpi=2000)

def return_sign(number):
    if number < 0.0:
        return '-'
    elif number == 0.0:
        return ''
    else:
        return '+'
    


def plot_loss_accuracy_with_epoch(loss, accuracy, outfile):
    COLOR_ACCURACY = "#69b3a2"
    COLOR_LOSS = "#3399e6"
    
    epoch_x = np.array(range(len(accuracy)))
    batchsize = np.around(len(loss)/len(accuracy))

    accuracy_x = epoch_x * batchsize
    loss_x = np.array(range(len(loss))) 
    
    fig = plt.figure(figsize=(24, 8))
    # 修改了ax的位置，将legend挤出图窗外，重现类似题主的问题
    ax1 = fig.add_axes([0.1, 0.2, 0.9, 0.9])
    #fig, ax1 = plt.subplots(figsize=(24, 8))
    ax2 = ax1.twinx()

    ax1.plot(loss_x, loss, color=COLOR_LOSS, lw=1)
    ax2.plot(accuracy_x, accuracy, color=COLOR_ACCURACY, lw=3)

    x_labels = [str(i) for i in epoch_x]
    ax1.set_xlabel("Epoch",fontsize=24)
    ax1.set_xticks(accuracy_x, x_labels, fontsize=20)
    ax1.set_ylabel("Loss", color=COLOR_LOSS, fontsize=24)
    ax1.tick_params(axis="y", labelcolor=COLOR_LOSS)
    ax1.tick_params(axis='y',labelsize=20)

    ax2.set_ylabel("R2", color=COLOR_ACCURACY, fontsize=24)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACCURACY)
    ax2.tick_params(axis='y',labelsize=20)


    fig.suptitle("Loss and R2 vs Epoch", fontsize=32)

    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    return                                                                                                                                                                                                                                                                                                                

