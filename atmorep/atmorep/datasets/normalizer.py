####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

import code
import numpy as np
import xarray as xr
import atmorep.config.config as config

######################################################
#                   Normalize                        #
######################################################

def normalize( data, norm, dates, year_base = 1979) :
  corr_data = np.array([norm[12*(dt.year-year_base) + dt.month-1] for dt in dates])
  mean, var = corr_data[:, 0], corr_data[:, 1]
  #print(f"[DEBUG] Normalizer: mean shape {mean.shape}, var shape {var.shape}, data shape {data.shape}")
  #print(f"[DEBUG] Normalizer: mean sample {mean.flatten()[:5]}, std sample {var.flatten()[:5]}")
  if (var == 0.).all():
    zeros = np.argwhere(var == 0.)
    nzeros = zeros.shape[0]
    print(f"Warning: var contains {nzeros} zero entries")
    maxmax_show = min(20, nzeros)
    for zi in range(maxmax_show):
      idx = tuple(int(x) for x in zeros[zi])
      # idx corresponds to positions inside `var` which starts with date index
      date_idx = idx[0] if len(idx) > 0 else None
      date_dt = dates[date_idx] if (date_idx is not None and date_idx < len(dates)) else None
      info = f"  zero #{zi+1}: indices(in var)={idx} date={date_dt}"
      # try best-effort interpretation by dimensionality
      try:
        if var.ndim == 5:
          # (n_dates, n_fields, n_levels, n_lats, n_lons)
          _, field_idx, level_idx, lat_idx, lon_idx = idx
          info += f" -> field_idx={field_idx} level={level_idx} lat={lat_idx} lon={lon_idx}"
        elif var.ndim == 4:
          # could be (n_dates, n_fields, n_lats, n_lons) or (n_dates, n_levels, n_lats, n_lons)
          info += f" -> remaining_indices={idx[1:]}"
        elif var.ndim == 3:
          # (n_dates, n_lats, n_lons)
          _, lat_idx, lon_idx = idx
          info += f" -> lat={lat_idx} lon={lon_idx}"
      except Exception:
        pass
      #print(info)
  if len(norm.shape) > 2 : #global norm
    return normalize_local(data, mean, var)
  else:
    return normalize_global( data, mean, var)
  
######################################################
def normalize_local( data, mean, var) :
  data = (data - mean) / var
  return data

######################################################
def normalize_global( data, mean, var) :
  for i in range( data.shape[0]) :
    data[i] = (data[i] - mean[i]) / var[i]
  return data


######################################################
#                  Denormalize                       #
######################################################
def denormalize(data, norm, dates, year_base = 1979) :
  corr_data = np.array([norm[12*(dt.year-year_base) + dt.month-1] for dt in dates])
  mean, var = corr_data[:, 0], corr_data[:, 1]
  if len(norm.shape) > 2 :
    return denormalize_local(data, mean, var)
  else:
    return denormalize_global(data, mean, var)  

######################################################

def denormalize_local(data, mean, var) :
  if len(data.shape) > 3: #ensemble
    for i in range( data.shape[0]) :
      data[i] = (data[i] * var) + mean
  else:
      data = (data * var) + mean
  return data

######################################################

def denormalize_global(data, mean, var) :
  if len(data.shape) > 3: #ensemble
    data = data.swapaxes(0,1)
    for i in range( data.shape[0]) :
      data[i] = ((data[i] * var[i]) + mean[i])
    data = data.swapaxes(0,1)
  else:
    for i in range( data.shape[0]) :
      data[i] = (data[i] * var[i]) + mean[i]
    
  return data