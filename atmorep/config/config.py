import os 
from pathlib import Path

fpath = os.path.dirname(os.path.realpath(__file__))
print("Using config file:", fpath)

path_models = Path('/work/ab1412/atmorep/models/') #Path( fpath, '../../models/')
path_results = Path('/work/ab1412/atmorep/results/') #Path( fpath, '../../results')
path_plots = Path('/work/ab1412/atmorep/results/plots/') #Path( fpath, '../results/plots/')

grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
                'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q',
                'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',
                'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                'total_precip' : 'tp', 'radar_precip' : 'yw_hourly',
                't2m' : 't_2m', 'u_10m' : 'u_10m', 'v_10m' : 'v_10m',  }
