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

from atmorep.core.evaluator import Evaluator
import time
import datetime

if __name__ == '__main__':

  model_id = 'iuw3ce3v'
  # 'j2l0sz9j' # repretrained model v8
  # 'iuw3ce3v' # t2m corrected v2.6

  # 'u9vvriz7' # v2.4
  # 'zxipahjj' # v1.2
  # 'ugqn2s9m' # v7
  # 'qw047nnt' # v5
  # '0rmiio09' # v4 of the fine-tuned t2m model
  # '58ipo6bs' # later version of fine-tuned t2m model
  # 'hjbmsjft' # first fine-tuned t2m model
  # 'wc5e2i3t' # 6 field OOTB
  # 'vkfqvtsq' # fine-tuned baseline forecast Arctic

  # model_id = 'tzs2378j'  # Bert Arctic, target precip and no masking of other fields.

  # arXiv 2023: models for individual fields
  #model_id = '4nvwbetz'     # vorticity
  #model_id = 'oxpycr7w'     # divergence
  #model_id = '1565pb1f'     # specific_humidity
  #model_id = '3kdutwqb'     # total precip
  #model_id = 'dys79lgw'     # velocity_u
  #model_id = '22j6gysw'     # velocity_v
  #model_id = '15oisw8d'     # velocity_z
  #model_id = '3qou60es'     # temperature 
  #model_id = '2147fkco'     # temperature (also 2147fkco)
  
  # new runs 2024
  #model_id='j8dwr5qj' #velocity_u
  #model_id='0tlnm5up' #velocity_v
  #model_id='v63l01zu' #specific humidity 
  #model_id='9l1errbo' #velocity_z
  #model_id='7ojls62c' #temperature 1024 
  
  # supported modes: test, forecast, fixed_location, temporal_interpolation, global_forecast,
  #                  global_forecast_range
  # options can be used to over-write parameters in config; some modes also have specific options, 
  # e.g. global_forecast where a start date can be specified

  #Add 'attention' : True to options to store the attention maps. NB. supported only for single field runs. 
  
  # BERT masked token model
  #mode, options = 'BERT', {'years_val' : [2015], 'num_samples_validate' : 96, 'with_pytest' : True}

  # BERT forecast mode
  #mode, options = 'forecast', {'forecast_num_tokens' : 2, 'num_samples_validate' : 128, 'with_pytest' : True }

  #temporal interpolation 
  #idx_time_mask: list of relative time positions of the masked tokens within the cube wrt num_tokens[0]  
  #mode, options = 'temporal_interpolation', {'idx_time_mask': [5,6,7], 'num_samples_validate' : 128, 'with_pytest' : True}

  # BERT forecast with patching to obtain global forecast
  # mode, options = 'global_forecast', { 
  #                                     #'dates' : [[2021, 2, 10, 12]]
  #                                     'dates' : [
  #                                        [2021, 1, 10, 12] , [2021, 1, 11, 0], [2021, 1, 11, 12], [2021, 1, 12, 0], #[2021, 1, 12, 12], [2021, 1, 13, 0], 
  #                                        [2021, 4, 10, 12], [2021, 4, 11, 0], [2021, 4, 11, 12], [2021, 4, 12, 0], #[2021, 4, 12, 12], [2021, 4, 13, 0], 
  #                                        [2021, 7, 10, 12], [2021, 7, 11, 0], [2021, 7, 11, 12], [2021, 7, 12, 0], #[2021, 7, 12, 12], [2021, 7, 13, 0], 
  #                                        [2021, 10, 10, 12], [2021, 10, 11, 0], [2021, 10, 11, 12], #[2021, 10, 12, 0], [2021, 10, 12, 12], [2021, 10, 13, 0]
  #                                      ], 
  #                                     'token_overlap' : [0, 0],
  #                                     'forecast_num_tokens' : 2,
  #                                     'with_pytest' : True }

  mode, options = 'global_forecast', { #'fields[0][2]' : [0],
                                     #'fields_prediction' : [['t2m', 1.0]],
                                     'geo_range_sampling' :  None, #[71, 1440],
                                     'dates' : [ [2021, 9, 26, 12] ],
                                     'token_overlap' : [0, 0],
                                     'forecast_num_tokens' : 2, 
                                     'attention' : False, 
                                     'with_pytest' : False}

  # mode, options = 'global_forecast_range', {
  #                                    #'dates' : [ [2021, 9, 26, 12] ], 
  #                                    'cur_date': [2015, 1, 22, 9],
  #                                    'token_overlap' : [0, 0],
  #                                    'geo_range_sampling' : None,
  #                                    'forecast_num_tokens' : 1,
  #                                    'attention' : False,
  #                                    'with_pytest' : False}


  # mode, options = 'fixed_location', {   'time_pos': ([[2015, 1, 21, 0, 81, 18.25], [2015, 1, 21, 1, 81, 18.25]]),
  #                                       'num_t_samples_per_month' : 1, 
  #                                       'years_val' : [2015]

  #                                       #'geo_range_sampling': [71, 1440], # now grid points for better overview
  #                                       #'days' : [21], # not sure this is accepted
  #                                       #'time' : [21] # not sure this is accepted either
  # }

  '''
  set_data [(self, times_pos, batch_size = None)]
      times_pos = np.array( [ [year, month, day, hour, lat, lon], ...]  )
        - lat \in [90,-90] = [90N, 90S]
        - lon \in [0,360]
        - (year,month) pairs should be a limited number since all data for these is loaded
  '''

# prototype for fixed location forecast
  # mode, options = 'fixed_location_forecast', {
  #                                       'pos': [82, 18.25],
  #                                       'years' : [2015],
  #                                       'months' : [1],
  #                                       'days' : [21], # not sure this is accepted
  #                                       'time' : [21] # not sure this is accepted either
  #                                   }

  file_path = '/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m.zarr'

  now = time.time()
  Evaluator.evaluate( mode, model_id, file_path, options)
  print("time", time.time() - now)


'''

To Do: 

1. evaluate only the samples from jan to june 2015 that are found in NICE. 
- fixed location 
- fixed time 
2. Use Dask to parallelise the evaluation.
3. compute the mean and standard deviation of the corrected t2m values for the evaluation period


'''
