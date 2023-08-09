import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import os

a=6371220 #radius of earth in m
L=2.5E6 #Latent heat in atmosphere
L_ice=3.34e5 #Latent heat of fusion
g=9.81 #Acceleration from gravity
conv_pw=1e15 #Conversion from watts to PW
cp=1007  

def calc_strm_funct(datas):
    '''Calculates the meridional overturning streamfunction from monthly data in pressure coordinates by month
    
    Args:
        datas(Xarray dataset)- An Xarray dataset from ERA5 output
        
    Output:
        strm_fnct_data(Xarray DataArray) - Meridional streamfunction
    '''
    
    time=datas.time
    lats=datas.latitude
    lons=datas.longitude
    levels=datas.level
    
    zon_norms = np.load('zonal_norms.npy') #Dims (level, lat, lon)
    
    #Divide weights by g to get the units right
    weights = np.load('aht_weights.npy')/ g #Dims (level, lat, lon)
    weights[np.isnan(weights)] = 0
    weights_zon_mean = np.nanmean(weights, axis=2) #Dims (level, lat)
    
    geom_multiplier = 2 * np.pi * a * np.cos(lats.values*np.pi/180) #Dims (lat)
    
    vcomp = datas.v #Dims (time, level, lat, lon)
    vcomp_zon_mean = np.nansum(vcomp * zon_norms[None,:,:,:], axis=3) #Dims (time, level, lat)

    mass_flux = vcomp_zon_mean * weights_zon_mean[None,:,:] * geom_multiplier[None,None,:] #Dims (time, level, lat)

    vcomp_baro = np.nansum(mass_flux, axis=1) / ((geom_multiplier)*np.nansum(weights_zon_mean, axis=0))[None,:] #Dims (time, lat)

    vcomp_corrected = vcomp_zon_mean - vcomp_baro[:, None,:] #Dims (time, level, lat)

    mass_flux_corrected = vcomp_corrected * weights_zon_mean[None,:,:] * geom_multiplier[None,None,:] #Dims (time, level, lat)
    mass_flux_corrected_reverse = mass_flux_corrected[:,::-1,:]
    
    strm_fnct = np.nancumsum(mass_flux_corrected, axis=1)

    strm_fnct_da = xr.DataArray(data=strm_fnct,
                                dims=['time', 'level', 'latitude'],
                            coords=dict(
                                time=time,
                                latitude=lats,
                                level=levels)
        )
        
    return(strm_fnct_da)



lats = np.linspace(90, -90, 361)
lons = np.linspace(0, 359.5, 720)
levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350,
            400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]


years = range(1979, 2019)
times = ['00', '06', '12', '18']

for year in years:
    for time in times:

        possible_file_name = f'strm_fncts/strm_fnct_{year}_{time}z.nc'
    
        #Check if file already exists
        if not os.path.isfile(possible_file_name):
            v_comp_data = xr.open_dataset(f'/home/disk/eos9/ERA5/hourly_pl/{time}/{year}.v.nc')
            strm_ds = calc_strm_funct(v_comp_data)
            
            strm_ds.to_netcdf(f'strm_fncts/strm_fnct_{year}_{time}z.nc')