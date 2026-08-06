import numpy as np
import xarray as xr
import zarr
import code
import pdb
from calendar import monthrange
import sys

#year = int(sys.argv[1])


def days_in_month( year, month) :
  '''days in month in specific year'''
  return monthrange( year, month)[1]

grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
              'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q', 
              'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',       
              'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
              'total_precip' : 'tp', 'radar_precip' : 'yw_hourly',
              't2m' : 't2m', 'u_10m' : 'u_10m', 'v_10m' : 'v_10m',  }

################################################
# set parameters

#era5
year_start = 2010 
year_end = 2021
#year_start = year_end = year
#fields = ['vorticity', 'divergence', 'velocity_u', 'velocity_v',  'velocity_z', 'temperature', 'specific_humidity']
fields = []
#levels = [ 96, 105, 114, 123, 137]
levels = []

#fields_sfc = ['total_precip'] #sst #geopotential  #total_precip
fields_sfc = ['t2m']

# control construction of zarr: full month data leads to out-of-memory error
t_delta = 6*24
num_lats, num_lons = 721, 1440  #era5
lat_min, lat_max = 0., 180.
lon_min, lon_max = 0., 360.
is_global = True # dataset is global covering the entire globe or not
# upsampling, i.e. reduce resolution
# set to 0 for none (num_lats etc has to match then); 2 for 1 deg 
upsampling_factor = 0 #2
flip_coordinates = False

################################################
# process parameters
#path_base = '/p/fastdata/slmet/slmet111/met_data/ecmwf/era5_reduced_level/ml_levels/{}/ml{}/'

#path_base = '/work/ab1412/atmorep/data/grib_files/{}/ml{}' 
#fname_base = path_base + '/era5_{}_y{}_m{:02d}_ml{}.grib'

path_base = '/work/ab1412/atmorep/data/grib_files/yearly_t2m_from_copernicus'
fname_base = path_base + '/era5_2m_temperature_{}.grib'

################################################
#helper functions

def reduce_resolution(data, upsampling_factor):
  for i_res in range( upsampling_factor) :
    lat_res = data.shape[1]
    data = 0.5 * (data[:,:,::2] + data[:,:,1::2])
    data[:,1:lat_res//2+1,:] = 0.5*(data[:,2::2,:] + data[:,1::2,:])
    data = data[:,:lat_res//2+1,:]
  return data

def get_data(field, fname, range, upsampling_factor):
  print(fname)
  ds_grib = xr.open_dataset(fname, 
                            engine='cfgrib',
                            backend_kwargs={'time_dims':('valid_time','indexing_time'),  "indexpath": ''} )

  fn = grib_index[ field ]
  data = ds_grib[ fn ].values[ range[0]:range[1] ]
  data = reduce_resolution(data, upsampling_factor)

  valid_time = ds_grib['valid_time'].values.astype('datetime64[s]')
  return data, valid_time

################################################
#main

if __name__ == "__main__":
  
  # TODO: adjust num_lats, num_lons based on upsampling_factor
  for i in range(upsampling_factor) :
    num_lons = num_lons // 2
    num_lats = ((num_lats-1) // 2) + 1
    t_delta *= 2

  print( f'num_lats / num_lons : {num_lats} / {num_lons}')

  ############
  # perfom computations
  res = [(lat_max - lat_min) / (num_lats-1),  (lon_max - lon_min) / num_lons]
  print(f"resolution: {res}")
  
  # append current chunk
  if year_start != year_end:
    #fname = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y{}_{}_res{:03d}_chunk8.zarr'.format(year_start, year_end, int(res[0]*100))
    #fname = '/work/ab1412/atmorep/data/combined/ml137/era5_y{}_{}_res{:03d}_chunk8.zarr'.format(year_start, year_end, int(res[0]*100))
    fname = "/work/ab1412/atmorep/data/t2m/era5_t2m_y{}_{}_res{:03d}_chunk8.zarr".format(year_start, year_end, int(res[0]*100))
  else:
    #fname = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y{}_res{:03d}_chunk8.zarr'.format(year_start, int(res[0]*100))
    #fname = '/work/ab1412/atmorep/data/combined/ml137/era5_y{}_res{:03d}_chunk8.zarr'.format(year_start, int(res[0]*100))
    fname = "/work/ab1412/atmorep/data/t2m/era5_y{}_res{:03d}_chunk8.zarr".format(year_start, int(res[0]*100))

  store = zarr.DirectoryStore(fname)
  print(fname)
  ds = zarr.group( store=store)
  ds.attrs['is_global']  = is_global
  ds.attrs['fields']     = fields
  ds.attrs['fields_sfc'] = fields_sfc
  ds.attrs['levels']     = levels
  ds.attrs['res']        = res
  
  ds.create_dataset( 'lats', data=np.linspace( lat_min, lat_max, num=num_lats, endpoint=True))
  ds.create_dataset( 'lons', data=np.linspace( lon_min, lon_max, num=num_lons, endpoint=False))
  ds.require_dataset( 'time', shape=(0,), dtype='datetime64[s]', chunks=(250))
 
  num_fields, num_levels = len(fields), len(levels)
  num_fields_sfc = len(fields_sfc)
  ds.require_dataset( 'data', shape=(0, num_fields, num_levels, num_lats, num_lons), dtype=np.float32,  chunks=( 8, num_fields, num_levels, num_lats, num_lons)) 
  ds.require_dataset( 'data_sfc', shape=(0, num_fields_sfc, num_lats, num_lons), dtype=np.float32, chunks=( 8, num_fields_sfc, num_lats, num_lons)) 
  
  ds_norm = ds.create_group( f'normalization' )
  ds_norm.require_dataset( 'norm', shape=(0, 2, len(fields), len(levels), num_lats, num_lons), dtype=np.float32, chunks=( 4, 2, 1, 1, num_lats, num_lons))
  ds_norm.require_dataset( 'norm_sfc', shape=(0, 2, len(fields_sfc), num_lats, num_lons), dtype=np.float32, chunks=( 4, 2, 1, num_lats, num_lons)) 
  ds_norm.require_dataset( 'global_norm', shape=(0, 2, len(fields), len(levels)), dtype=np.float32, chunks=( 4, 2, 1, 1)) 
  ds_norm.require_dataset( 'global_norm_sfc',shape=(0, 2, len(fields_sfc)), dtype=np.float32, chunks=( 4, 2, 1)) 
 
  for year in range( year_start, year_end+1) :
    for month in range( 1, 12+1) :
    
      print('year {} - month {:02d}'.format(year, month))
      
      #TODO: add check to see if the folder exists
      
      t_steps = 24 * days_in_month( year, month)  
      
      temp     = np.zeros([len(fields), len(levels), t_steps, num_lats, num_lons], dtype = np.float32)
      temp_sfc = np.zeros([len(fields_sfc), t_steps, num_lats, num_lons], dtype = np.float32)

      norm     = np.zeros([2, len(fields),len(levels), num_lats, num_lons], dtype = np.float32)
      norm_sfc = np.zeros([2, len(fields_sfc), num_lats, num_lons], dtype = np.float32)
      
      global_norm     = np.zeros([2, len(fields), len(levels)], dtype = np.float32)
      global_norm_sfc = np.zeros([2, len(fields_sfc)], dtype = np.float32)

      for ilvl, level in enumerate(levels) :
        for ifld, field in enumerate(fields) :
          print(f"level: {level} - field: {field}")
          ds_grib = xr.open_dataset( fname_base.format(year), #fname_base.format( field, level, field, year, month, level), 
                                      engine='cfgrib',
                                      backend_kwargs={'time_dims':('valid_time','indexing_time'),  "indexpath": ''} )
          time = ds_grib['valid_time'].values[:t_steps].astype('datetime64[s]')
          fn = grib_index[ field ]
          data = ds_grib[ fn ].values[ : t_steps]
          data = reduce_resolution(data, upsampling_factor)
          global_norm[0, ifld, ilvl] = np.mean(data)
          global_norm[1, ifld, ilvl] = np.std(data)
          temp[ifld, ilvl]    = data
          norm[0, ifld, ilvl] = data.mean(axis = 0)
          norm[1, ifld, ilvl] = data.std(axis = 0)

      #surface data
      for ifld_sfc, field_sfc in enumerate(fields_sfc) :
        print(f"surface field: {field_sfc}")
        ds_grib = xr.open_dataset( fname_base.format(year), #fname_base.format(field_sfc, 0, field_sfc, year, month, 0), 
                                      engine='cfgrib',
                                      backend_kwargs={'time_dims':('valid_time','indexing_time'),  "indexpath": ''} )

        fn = grib_index[ field_sfc ] 
        data_sfc = ds_grib[fn].values[:t_steps]
        time = ds_grib['valid_time'].values[:t_steps].astype('datetime64[s]') 
        #data_sfc = ds_grib[ fn ].values[  : t_steps ]
        data_sfc = reduce_resolution(data_sfc, upsampling_factor)
        global_norm_sfc[0, ifld_sfc] = np.mean(data_sfc)
        global_norm_sfc[1, ifld_sfc] = np.std(data_sfc)
        
        temp_sfc[ifld_sfc]    = data_sfc
        norm_sfc[0, ifld_sfc] = data_sfc.mean(axis = 0)
        norm_sfc[1, ifld_sfc] = data_sfc.std(axis = 0)
      
      ds['time'].append( time , axis = 0 )
      #ds['data'].append( temp.transpose(2,0,1,3,4), axis=0)
      #surface fields
      ds['data_sfc'].append(temp_sfc.transpose(1,0,2,3), axis = 0)
      #ds['normalization/norm'].append( np.expand_dims(norm, axis=0) , axis=0)
      #surface fields
      ds['normalization/norm_sfc'].append(np.expand_dims(norm_sfc, axis=0), axis = 0)
      #ds['normalization/global_norm'].append( np.expand_dims(global_norm, axis=0), axis=0)
      #surface fields
      ds['normalization/global_norm_sfc'].append(np.expand_dims(global_norm_sfc, axis=0), axis = 0)
      
    # added to ignore if i dont have 3D data 
    # flip coordinates if required# Only append if there are 3D fields/levels
      if len(fields) > 0 and len(levels) > 0:
          ds['data'].append(temp.transpose(2,0,1,3,4), axis=0)
          ds['normalization/norm'].append(np.expand_dims(norm, axis=0), axis=0)
          ds['normalization/global_norm'].append(np.expand_dims(global_norm, axis=0), axis=0)
      
      
      print( f'Finished {month}/{year}.')


    # not yet implemented
    if flip_coordinates:
      ds["lats"] = 90 - ds["lats"]
    
    store.close()
    print( f'Finished {year}.')
    
