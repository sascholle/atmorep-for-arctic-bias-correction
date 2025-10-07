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

import torch
import os
import sys
import traceback
import pdb
import wandb
import zarr

from atmorep.core.trainer import Trainer_BERT
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch

from torchinfo import torchinfo


####################################################################################################
def train() :

  devices = init_torch()
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  # torch.cuda.set_sync_debug_mode(1)
  torch.backends.cuda.matmul.allow_tf32 = True

  cf = Config()
  # parallelization
  cf.with_ddp = with_ddp
  cf.num_accs_per_task = len(devices)   # number of GPUs / accelerators per task
  cf.par_rank = par_rank
  cf.par_size = par_size
  
  # format: list of fields where for each field the list is 
  # [ name , 
  #   [ dynamic or static field { 1, 0 }, embedding dimension, , device id ],
  #   [ vertical levels ],
  #   [ num_tokens],
  #   [ token size],
  #   [ total masking rate, rate masking, rate noising, rate for multi-res distortion]
  # ]

  cf.fields = [
    [
        'vorticity',  # Name
        [1, 1024, [], 0],  # Field Properties: [Dynamic, Embedding Dimension, Device ID]
        [137],  # Vertical Levels (highlighted code)
        [12, 3, 6],  # Number of Tokens
        [3, 18, 18],  # Token Size
        [0.5, 0.9, 0.2, 0.05]  # Masking and Noising Rates
    ]
]

  # cf.fields = [ [ 'temperature', [ 1, 1024, [ ], 0 ], 
  #                              [ 96, 105, 114, 123, 137 ], 
  #                              [12, 2, 4], [3, 27, 27], [0.5, 0.9, 0.2, 0.05], 'local' ] ]
  cf.fields_prediction = [ [cf.fields[0][0], 1.] ]
 
  # cf.fields = [ [ 'velocity_u', [ 1, 1024, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                                [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ] ]

  # cf.fields_prediction = [ [cf.fields[0][0], 1.] ]

  
  # cf.fields = [ [ 'velocity_v', [ 1, 1024, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 3, 6], [3, 18, 18], [0.25, 0.9, 0.1, 0.05] ] ]

  # cf.fields = [ [ 'velocity_z', [ 1, 1024, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 3, 6], [3, 18, 18], [0.25, 0.9, 0.1, 0.05] ] ]

  # cf.fields = [ [ 'specific_humidity', [ 1, 1024, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 3, 6], [3, 18, 18], [0.25, 0.9, 0.1, 0.05] ] ]
  
  cf.fields_targets = []

  #cf.years_train = list( range( 1979, 2021))
  cf.years_train = list( range(2021, 2022))
  cf.years_train = []
  cf.years_val = [2021]  #[2018] 
  cf.month = None
  cf.geo_range_sampling = [[ -90., 90.], [ 0., 360.]]
  cf.time_sampling = 1   # sampling rate for time steps
  # random seeds
  cf.torch_seed = torch.initial_seed()
  # training params
  cf.batch_size_validation = 1 #64
  cf.batch_size = 96
  cf.num_epochs = 400 #128
  cf.num_samples_per_epoch = 4096*12
  cf.num_samples_validate = 128*12
  cf.num_loader_workers = 8
  
  # additional infos
  cf.size_token_info = 8
  cf.size_token_info_net = 16
  cf.grad_checkpointing = True
  cf.with_cls = False
  # network config
  cf.with_mixed_precision = True
  cf.with_layernorm = True
  cf.coupling_num_heads_per_field = 1
  cf.dropout_rate = 0.05
  cf.with_qk_lnorm = False
  # encoder
  cf.encoder_num_layers = 6
  cf.encoder_num_heads = 16
  cf.encoder_num_mlp_layers = 2
  cf.encoder_att_type = 'dense'
  # decoder
  cf.decoder_num_layers = 6
  cf.decoder_num_heads = 16
  cf.decoder_num_mlp_layers = 2
  cf.decoder_self_att = False
  cf.decoder_cross_att_ratio = 0.5
  cf.decoder_cross_att_rate = 1.0
  cf.decoder_att_type = 'dense'
  # tail net
  cf.net_tail_num_nets = 16
  cf.net_tail_num_layers = 0
  # loss
  cf.losses = ['mse_ensemble'] # mse, mse_ensemble, stats, crps, weighted_mse
  # training
  cf.optimizer_zero = False
  cf.lr_start = 5. * 10e-7
  cf.lr_max = 0.00005*3
  cf.lr_min = 0.00004 #0.00002
  cf.weight_decay = 0.05 #0.1
  cf.lr_decay_rate = 1.025
  cf.lr_start_epochs = 3
  cf.model_log_frequency = 256 #save checkpoint every X batches

  # BERT
  # strategies: 'BERT', 'forecast', 'temporal_interpolation'
  cf.BERT_strategy = 'forecast'
  cf.forecast_num_tokens = 2      # only needed / used for BERT_strategy 'forecast
  cf.BERT_fields_synced = False   # apply synchronized / identical masking to all fields 
                                  # (fields need to have same BERT params for this to have effect)
  cf.BERT_mr_max = 2              # maximum reduction rate for resolution
  
  # debug / output
  cf.log_test_num_ranks = 0
  cf.save_grads = False
  cf.profile = False
  cf.test_initial = False
  cf.attention = False

  cf.rng_seed = None 

  # usually use %>wandb offline to switch to disable syncing with server
  cf.with_wandb = True
  setup_wandb( cf.with_wandb, cf, par_rank, 'train', mode='offline')  

  # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res100_chunk32.zarr'
  # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res100_chunk32.zarr'
  # # # in steps x lat_degrees x lon_degrees
  # cf.n_size = [36, 1*9*6, 1.*9*12]

  # # # # # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res025_chunk16.zarr'
  # # # # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res025_chunk32.zarr'
  # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res025_chunk32.zarr'
  # # # 
  # # # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res025_chunk8.zarr'
  # # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res025_chunk8_lat180_lon180.zarr'
  # # # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res025_chunk16.zarr'
  #cf.file_path = '/gpfs/scratch/ehpc03/era5_y1979_2021_res025_chunk8.zarr/'
  cf.file_path = '/work/ab1412/atmorep/data/vorticity/ml137/era5_y2021_res025_chunk8.zarr'
  
  # # # in steps x lat_degrees x lon_degrees
  cf.n_size = [36, 0.25*9*6, 0.25*9*12]

  # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res100_chunk16.zarr'
  #cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res100_chunk16.zarr'
  #cf.n_size = [36, 1*9*6, 1.*9*12]

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  trainer = Trainer_BERT( cf, devices).create()
  trainer.run()


####################################################################################################


def train_continue( wandb_id, epoch, Trainer, epoch_continue = -1) :

  devices = init_torch()
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  cf = Config().load_json( wandb_id) # load model and point to file path for data

  cf.num_accs_per_task = 1   # number of GPUs / accelerators per task
  cf.with_ddp = with_ddp
  cf.par_rank = par_rank
  cf.par_size = par_size
  cf.optimizer_zero = False
  cf.attention = False

  # name has changed but ensure backward compatibility
  if hasattr( cf, 'loader_num_workers') :
    cf.num_loader_workers = cf.loader_num_workers
  if not hasattr( cf, 'n_size'):
    cf.n_size = [36, 0.25*9*6, 0.25*9*12] # in steps x lat_degrees x lon_degrees
  if not hasattr(cf, 'num_samples_per_epoch'):
    cf.num_samples_per_epoch = 1024
  if not hasattr(cf, 'num_samples_validate'):
    cf.num_samples_validate = 128
  if not hasattr(cf, 'with_mixed_precision'):
    cf.with_mixed_precision = True
  if not hasattr(cf, 'years_val'):
    cf.years_val = cf.years_test
    

  # any parameter in cf can be overwritten when training is continued, e.g. we can increase the 
  # masking rate 


######################################################
# Parameters changed by me
######################################################

# for 6 field model

  cf.fields = [
    [
        'velocity_u',  # Name
        [1, 1024, ['velocity_v', 'temperature'], 0, ['j8dwr5qj', -2]],  # Field Properties # now 1024 / otherwise 2048
        [96, 105, 114, 123, 137],  # Vertical Levels
        [12, 3, 6],  # Number of Tokens
        [3, 18, 18],  # Token Size
        #[12, 6, 12],
        #[3, 9, 9],
        #[0.1, 0, 0, 0] 
        [0.5, 0.9, 0.2, 0.05] # [ total masking rate, rate masking, rate noising, rate for multi-res distortion]
    ],
    [
        'velocity_v',  # Name 
        [1, 1024, ['velocity_u', 'temperature'], 1, ['0tlnm5up', -2]],  # Field Properties # 1024 / 2048
        [96, 105, 114, 123, 137],  # Vertical Levels
        [12, 3, 6],  # Number of Tokens
        [3, 18, 18],  # Token Size 
        #[12, 6, 12],
        #[3, 9, 9],
        #[0.1, 0, 0, 0] 
        [0.5, 0.9, 0.2, 0.05]  # Masking and Noising Rates
    ],
    [
        'specific_humidity',  # Name
        [1, 1024, ['velocity_u', 'velocity_v', 'temperature'], 2, ['v63l01zu', -2]],  # Field Properties #  1024  /2048
        [96, 105, 114, 123, 137],  # Vertical Levels
        [12, 3, 6],  # Number of Tokens
        [3, 18, 18],  # Token Size
        #[12, 6, 12],
        #[3, 9, 9],
        #[0.1, 0, 0, 0] 
        [0.5, 0.9, 0.2, 0.05]  # Masking and Noising Rates
    ],
    [
        'velocity_z',  # Name
        [1, 1024, ['velocity_u', 'velocity_v', 'temperature'], 3, ['9l1errbo', -2]],  # Field Properties 
        [96, 105, 114, 123, 137],  # Vertical Levels
        [12, 3, 6],  # Number of Tokens
        [3, 18, 18],  # Token Size
        #[12, 6, 12],
        #[3, 9, 9],
        #[0.1, 0, 0, 0] 
        [0.5, 0.9, 0.2, 0.05]  # Masking and Noising Rates
    ],
    [
        'temperature',  # Name
        [1, 1024, ['velocity_u', 'velocity_v', 'specific_humidity'], 3, ['7ojls62c', -2]],  # Field Properties # 1024 / 1536
        [96, 105, 114, 123, 137],  # Vertical Levels
        [12, 2, 4],  # Number of Tokens
        [3, 27, 27],  # Token Size
        #[0.1, 0, 0, 0],
        [0.5, 0.9, 0.2, 0.05],  # Masking and Noising Rates
        'Local' #Norm
    ],
    [
        'total_precip',  # Name
        [1, 1024, ['velocity_u', 'velocity_v', 'velocity_z', 'specific_humidity'], 0],  # Field Properties # was 1024 / 1536
        [0],  # Vertical Levels
        [12, 6, 12],  # Number of Tokens
        [3, 9, 9],  # Token Size
        #[0.1, 0, 0, 0] # Masking and Noising Rates
        [0.25,0.9,0.1,0.05]
    ],
    [
        't2m',  # Name
        [1, 1024, ['velocity_u', 'velocity_v', 'velocity_z', 'specific_humidity', 'temperature'], 1],  # Field Properties # was 1024 / 1536
        [0],  # Vertical Levels
        [12, 2, 4],  # Number of Tokens
        [3, 27, 27],  # Token Size
        #[1, 0, 0, 0], # Masking and Noising Rates
        [0.5, 0.9, 0.2, 0.05], 
        'Local'
    ], 
    [
        'corrected_t2m',  # Name
        [1, 1024, ['velocity_u', 'velocity_v', 'velocity_z', 'specific_humidity', 't2m'], 2],  # Field Properties # was 1024 / 1536
        [0],  # Vertical Levels
        [12, 2, 4],  # Number of Tokens
        [3, 27, 27],  # Token Size
        #[1, 0, 0, 0], # Masking and Noising Rates
        [0.1, 0.9, 0.2, 0.05], 
        'Local' 
    ]
  ]

  cf.fields_prediction = [

  #["velocity_u",0.225],["velocity_v",0.225],["specific_humidity",0.15],["velocity_z",0.1],["temperature",0.2],["total_precip",0.1] ]

  ["velocity_u", 0.125], ["velocity_v", 0.125], ["specific_humidity", 0.05], ["velocity_z", 0.01], ["temperature", 0.1], ["total_precip", 0.01], ["t2m", 0.2], ["corrected_t2m", 0.38] ]

  #cf.fields_targets = ["t2m"]
  cf.losses = ['mse_ensemble', 'stats'] # mse, mse_ensemble, stats, crps, weighted_mse

 # target sparsity section
  # cf.sparse_target = True  # Enable sparse target masking - only necessary for forecasting 
  # cf.sparse_target_field = 't2m'  # Field to apply sparsity to
  # cf.sparse_target_sparsity = 0.9  # ratio of data that will be masked

  # mask input field section
  # cf.mask_input_field = 't2m'  # or True to use default 'total_precip'
  # cf.mask_input_value = 0 # mask value, NaN is default 

  cf.batch_size = 64
  cf.num_loader_workers = 5
  cf.num_samples_per_epoch = 480 #1024 train continue 4096*12 train 1024 and I have an OOM error 
  cf.num_samples_validate = 128 #128 train continue 128*12 train
  cf.num_epochs = 128 # 400 / 128
  
  if not hasattr(cf, 'batch_size_validation'):
    cf.batch_size_validation = 1
  if not hasattr(cf, 'model_log_frequency'):
    cf.model_log_frequency = 256 #save checkpoint every X batches
  if not hasattr(cf, 'forecast_num_tokens'):
    cf.forecast_num_tokens = 2 #  only needed / used for BERT_strategy 'forecast'

  cf.BERT_strategy = 'BERT' #'forecast' #BERT
  cf.years_train = list( range(2010, 2021))
  cf.years_test = [2021]
  cf.years_val = [2021] 
  #cf.geo_range_sampling = [[ 72.27, 90.], [ 0., 360.]] #[[ -90., 90.], [ 0., 360.]]
  cf.geo_range_sampling = [list(range(0, 71)), list(range(0, 1440))]

  cf.file_path = '/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m.zarr' 
  #cf.file_path = "/work/ab1412/atmorep/data/era5_y2010_2020_res25_with_t2m.zarr"
  
  setup_wandb( cf.with_wandb, cf, par_rank, project_name='train', mode='offline')  

  # resuming a run requires online mode, which is not available everywhere
  #setup_wandb( cf.with_wandb, cf, par_rank, wandb_id = wandb_id) 
  
  if cf.with_wandb and 0 == cf.par_rank :
    #print if write_json is working
    cf.write_json( wandb)
    cf.print() 

  if -1 == epoch_continue :
    epoch_continue = epoch
  
  # output/debug
  cf.log_test_num_ranks = 0


  # run
  trainer = Trainer.load( cf, wandb_id, epoch, devices)
  print( 'Loaded run \'{}\' at epoch {}.'.format( wandb_id, epoch))

  trainer.run( epoch_continue)

####################################################################################################

if __name__ == '__main__':
  
  try :

    #train()

    # corrected training 
    # 0: Wandb run: atmorep-eg3ztaai-19017100 on b9h8xdoz v1.0.1
    # 0: Wandb run: atmorep-zxipahjj-19062614 on ugqn2s9m v1.0.2 - ugqn2s9m is an b9h8xdoz 6 field model 
    
    # second round
    # 0: Wandb run: atmorep-6mb1bcla-19305085 trained on j2l0sz9j 0.9 masking v1.1.1
    # 0: Wandb run: atmorep-qb7ksr0h-19309940 trained on 6mb1bcla 0.5 masking v1.1.2
    # 0: Wandb run: atmorep-y1gpdgaa-19313842 trained on qb7ksr0h 0.5 masking v1.1.3
    # 0: Wandb run: atmorep-u9vvriz7-19337083 trained on y1gpdgaa 0.5 masking v1.1.4
    # 0: Wandb run: atmorep-iuy5bnth-19366232 trained on u9vvriz7 0.1 masking v1.1.5
    # 0: Wandb run: atmorep-iuw3ce3v-19413611 trained on iuy5bnth 0.1 masking v1.1.6



    wandb_id, epoch, epoch_continue = 'iuy5bnth', -2, -2 
    Trainer = Trainer_BERT  
    train_continue( wandb_id, epoch, Trainer, epoch_continue)

  except :
    
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

