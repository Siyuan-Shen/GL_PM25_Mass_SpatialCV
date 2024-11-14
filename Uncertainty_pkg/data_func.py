import numpy as np 
import os
from scipy.interpolate import NearestNDInterpolator
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess
import time
from sklearn.neighbors import BallTree

from Training_pkg.iostream import load_monthly_obs_data
from Training_pkg.utils import *

from Estimation_pkg.data_func import get_landtype,get_extent_index
from Estimation_pkg.utils import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.iostream import *

from Uncertainty_pkg.iostream import load_GL_GeoLatLon,load_GL_GeoLatLon_Map,save_nearest_site_distances_forEachPixel
from Uncertainty_pkg.utils import *


def Get_LOWESS_values_for_Uncertainty(total_channel_names,width,height):
    nchannel = len(total_channel_names)
    SPECIES_OBS, sitelat, sitelon  = load_monthly_obs_data(species)
    total_obs_data = {}
    total_final_data = {}
    total_nearest_distances_data = {}
    total_nearbysites_distances_data = {}

    init_bins = np.linspace(0,Max_distances_for_Bins,Number_of_Bins)
    output_bins = (np.array(range(len(init_bins)-1))*round(Max_distances_for_Bins/Number_of_Bins))
    LOWESS_values = {}
    rRMSE         = {}
    Keys = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual','MAM','JJA','SON','DJF']
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    SEASONS = ['MAM','JJA','SON','DJF']
    MONTHS_inSEASONS = [['Mar', 'Apr', 'May'],[ 'Jun', 'Jul', 'Aug'],['Sep', 'Oct', 'Nov'],['Dec','Jan', 'Feb']]

    YEARS = [str(iyear) for iyear in range(Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear+1)]
    typeName = Get_typeName(bias=bias,normalize_bias=normalize_bias,normalize_species=normalize_species,absolute_species=absolute_species,log_species=log_species,species=species)
    for ikey in Keys:
        LOWESS_values[ikey] = np.array([],dtype=np.float64)
        rRMSE[ikey] = np.array([],dtype=np.float64)
        total_obs_data[ikey] = np.array([],dtype=np.float64)
        total_final_data[ikey] = np.array([],dtype=np.float64)
        total_nearest_distances_data[ikey] = np.array([],dtype=np.float64)
        total_nearbysites_distances_data[ikey] = np.array([],dtype=np.float64)
    for radius in Uncertainty_Buffer_radii_forUncertainty:
        temp_distances_recording = np.array([],dtype=np.float64)
        obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording,testsites2trainsites_nearest_distances,test_sites_index_recording,train_sites_index_recording,excluded_sites_index_recording, train_index_number, test_index_number=load_month_based_BLCO_data_recording(species, version, typeName, Uncertainty_BLISCO_beginyear, Uncertainty_BLISCO_endyear, nchannel, special_name, width, height,radius,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers)
        for ifold in range(Uncertainty_BLISCO_kfolds):
            for isite in range(len(test_sites_index_recording[str(ifold)])):
                test_index =  test_sites_index_recording[str(ifold)][isite]
                train_index = train_sites_index_recording[str(ifold)]
                temp_nearbysites_distance =  Get_NearbySites_Distances_Info(sitelat[test_index],sitelon[test_index],sitelat[train_index],sitelon[train_index],number_of_nearby_sites_forAverage,nearby_sites_distances_mode)
                temp_distances_recording = np.append(temp_distances_recording,temp_nearbysites_distance)
        for iyear in YEARS:
            for imonth in MONTHS:
                if imonth == 'Jan':
                    obs_data[iyear]['Annual'] = obs_data[iyear][imonth].copy()
                    final_data[iyear]['Annual'] = final_data[iyear][imonth].copy()
                else:
                    obs_data[iyear]['Annual'] += obs_data[iyear][imonth].copy()
                    final_data[iyear]['Annual'] += final_data[iyear][imonth].copy()
                total_obs_data[imonth] = np.append(total_obs_data[imonth],obs_data[iyear][imonth])
                total_final_data[imonth] = np.append(total_final_data[imonth],final_data[iyear][imonth])
                total_nearbysites_distances_data[imonth] = np.append(total_nearbysites_distances_data[imonth],temp_distances_recording)
                total_nearest_distances_data[imonth] = np.append(total_nearest_distances_data[imonth],testsites2trainsites_nearest_distances)
            for iseason in range(len(SEASONS)):
                for imonth in range(len(MONTHS_inSEASONS[iseason])):
                    if imonth == 0:
                        obs_data[iyear][SEASONS[iseason]]   = obs_data[iyear][MONTHS_inSEASONS[iseason][imonth]].copy()
                        final_data[iyear][SEASONS[iseason]] = final_data[iyear][MONTHS_inSEASONS[iseason][imonth]].copy()
                    else:
                        obs_data[iyear][SEASONS[iseason]]   += obs_data[iyear][MONTHS_inSEASONS[iseason][imonth]].copy()
                        final_data[iyear][SEASONS[iseason]] += final_data[iyear][MONTHS_inSEASONS[iseason][imonth]].copy()
                obs_data[iyear][SEASONS[iseason]] = obs_data[iyear][SEASONS[iseason]]/3.0
                final_data[iyear][SEASONS[iseason]] = final_data[iyear][SEASONS[iseason]] /3.0
                total_obs_data[SEASONS[iseason]] = np.append(total_obs_data[SEASONS[iseason]],obs_data[iyear][SEASONS[iseason]])
                total_final_data[SEASONS[iseason]] = np.append(total_final_data[SEASONS[iseason]],final_data[iyear][SEASONS[iseason]])
                total_nearbysites_distances_data[SEASONS[iseason]] = np.append(total_nearbysites_distances_data[SEASONS[iseason]],temp_distances_recording)
                total_nearest_distances_data[SEASONS[iseason]] = np.append(total_nearest_distances_data[SEASONS[iseason]],testsites2trainsites_nearest_distances)

            obs_data[iyear]['Annual'] = obs_data[iyear]['Annual']/12.0
            final_data[iyear]['Annual'] =  final_data[iyear]['Annual']/12.0
            total_obs_data['Annual'] = np.append(total_obs_data['Annual'],obs_data[iyear]['Annual'])
            total_final_data['Annual'] = np.append(total_final_data['Annual'],final_data[iyear]['Annual'])
            total_nearbysites_distances_data['Annual'] = np.append(total_nearbysites_distances_data['Annual'],temp_distances_recording)
            total_nearest_distances_data['Annual'] = np.append(total_nearest_distances_data['Annual'],testsites2trainsites_nearest_distances)
    
    for imonth in Keys:
        distances = total_nearbysites_distances_data[imonth].copy()
        temp_obs  = total_obs_data[imonth].copy()
        temp_final= total_final_data[imonth].copy()
        number_each_bin = np.array([],dtype=np.float64)
        for i in range(len(init_bins)-1):
            index = np.where(distances<init_bins[i+1])[0]
            temp_rRMSE = Cal_NRMSE_forUncertainty_Bins(temp_final[index],temp_obs[index],Low_percentile_remove,High_percentile_remove)
            rRMSE[imonth] = np.append(rRMSE[imonth],temp_rRMSE)
            distances = np.delete(distances,index)
            temp_final= np.delete(temp_final,index)
            temp_obs  = np.delete(temp_obs,index)
            number_each_bin = np.append(number_each_bin,len(index))
        temp_lowess_result = lowess(rRMSE[imonth], output_bins, frac=LOWESS_frac)
        smoothed_x = temp_lowess_result[:, 0]
        smoothed_y = temp_lowess_result[:, 1]
        LOWESS_values[imonth] = smoothed_y
        
    return LOWESS_values,rRMSE,output_bins

def Get_NearbySites_Distances_Info(test_lat,test_lon,train_lat_array,train_lon_array,number_of_nearby_sites_forAverage,nerby_sites_distances_mode='mean'):
    dist_map = calculate_distance_forArray(test_lat,test_lon,train_lat_array,train_lon_array)
    dist_map.sort()
    if  nerby_sites_distances_mode == 'mean':
        distance = np.mean(dist_map[0:number_of_nearby_sites_forAverage])
    if nerby_sites_distances_mode == 'median':
        if number_of_nearby_sites_forAverage%2 == 1:
            distance = dist_map[int((number_of_nearby_sites_forAverage-1)/2)]
        else:
            distance = np.mean(dist_map[int(number_of_nearby_sites_forAverage)/2-1:int(number_of_nearby_sites_forAverage/2)+1])
    return distance
    
def Cal_NRMSE_forUncertainty_Bins(final_data,obs_data,low_percentile,high_percentile):
    ratio = final_data/obs_data
    percentage_array = np.array(range(21))*5
    low_percentile_index = int(low_percentile/5.0)
    high_percentile_index = int(high_percentile/5.0)
    threshold_array = np.percentile(ratio,percentage_array)
    ratio_forCalculation_index = np.where((ratio>=threshold_array[low_percentile_index])&(ratio<=threshold_array[high_percentile_index]))
    #RMSE = np.sqrt(mean_squared_error(ratio[ratio_forCalculation_index], ratio[ratio_forCalculation_index]/ratio[ratio_forCalculation_index]))
    #RMSE = np.sqrt(mean_squared_error(final_data[ratio_forCalculation_index]/obs_data[ratio_forCalculation_index], obs_data[ratio_forCalculation_index]/obs_data[ratio_forCalculation_index]))
    RMSE = np.sqrt(mean_squared_error(final_data[ratio_forCalculation_index],obs_data[ratio_forCalculation_index]))
    #RMSE = np.sqrt(mean_squared_error(final_data,obs_data))
    
    RMSE = round(RMSE, 2)
    
    #NRMSE = RMSE
    NRMSE = RMSE/np.mean(obs_data[ratio_forCalculation_index])
    #NRMSE = RMSE/np.mean(obs_data)
    return NRMSE

def convert_distance_to_rRMSE_uncertainty(distances_bins_array, BLCO_rRMSE_LOWESS_values, map_distances,):
    print('Get into the convert_distance_to_rRMSE_uncertainty!!!!!')
    map_uncertainty = np.zeros(map_distances.shape,dtype=np.float64)
    pixels_index = np.where(map_distances < distances_bins_array[0])
    map_uncertainty[pixels_index] = BLCO_rRMSE_LOWESS_values[0]
    for iradius in range(len(distances_bins_array)-1):
        d_left  = distances_bins_array[iradius]
        d_right = distances_bins_array[iradius+1]
        rRMSE_left  = BLCO_rRMSE_LOWESS_values[iradius]
        rRMSE_right = BLCO_rRMSE_LOWESS_values[iradius+1]
        pixels_index = np.where((map_distances >= d_left) & (map_distances < d_right))
        print('d_left: {}, d_right: {}, rRMSE_left: {}, rRMSE_right: {}'.format(d_left,d_right,rRMSE_left,rRMSE_right))
        map_uncertainty[pixels_index] = (map_distances[pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
        #print('Distance: ', map_distances[pixels_index], 'Uncertainty: ',map_uncertainty[pixels_index])
    
    d_left  = distances_bins_array[0]
    d_right = distances_bins_array[-1]
    rRMSE_left  = BLCO_rRMSE_LOWESS_values[0]
    rRMSE_right = BLCO_rRMSE_LOWESS_values[-1] 
    outrange_pixels_index = np.where(map_distances >= distances_bins_array[-1])
    if BLCO_rRMSE_LOWESS_values[-1] >= BLCO_rRMSE_LOWESS_values[-2]:
        slope = abs(BLCO_rRMSE_LOWESS_values[-1]-BLCO_rRMSE_LOWESS_values[-2])/(distances_bins_array[-1]-distances_bins_array[-2])
        map_uncertainty[outrange_pixels_index] = slope*(map_distances[outrange_pixels_index]-distances_bins_array[-1])+BLCO_rRMSE_LOWESS_values[-1]
    else:
        map_uncertainty[outrange_pixels_index] = rRMSE_right #(map_distances[outrange_pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
    
    return map_uncertainty

def get_nearest_site_distance_for_each_pixel():
    SATLAT,SATLON = load_GL_GeoLatLon()
    lat_index, lon_index = get_extent_index(Extent)
    extent_lat_map,extent_lon_map = get_extent_lat_lon_map(lat_index=lat_index,lon_index=lon_index,SATLAT=SATLAT,SATLON=SATLON)
    SPECIES, sites_lat,sites_lon = load_monthly_obs_data(species=species)
    sites_loc = np.array([sites_lat,sites_lon]).T
    sites_loc = np.radians(sites_loc)
    tree = BallTree(sites_loc, metric='haversine',leaf_size=2) # build ball tree
    landtype = get_landtype(YYYY=2015,extent=Extent)
    nearest_distance_map = np.full(extent_lat_map.shape,1000.0)
    for ix in range(len(lat_index)):
        land_index = np.where(landtype[ix,:] != 0)
        print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
        if len(land_index[0]) == 0:
            print('No lands.')
            None
        else:
            start_time = time.time()
            temp_pixels_lat_lon = np.radians(np.array([extent_lat_map[ix,land_index[0]], extent_lon_map[ix,land_index[0]]]).T)
            dist, ind = tree.query(temp_pixels_lat_lon,k=1)  # find sites nearby
            dist = dist * 6371 # convert to km
            distances = np.squeeze(dist)
            nearest_distance_map[ix,land_index[0]] = distances
            end_time = time.time()
            Get_distance_forOneLatitude_time = end_time - start_time
            print('Time for getting distance for one latitude', Get_distance_forOneLatitude_time, 's, the number of pixels is ', len(land_index[0]))

    save_nearest_site_distances_forEachPixel(nearest_distance_map=nearest_distance_map,extent_lat=SATLAT[lat_index],extent_lon=SATLON[lon_index])
    return 
