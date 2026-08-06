# AtmoRep

This repository contains the source code for fine-tuning the [AtmoRep](https://www.atmorep.org) models for Arctic t2m bias-correction as part of an internship completed in 2025 at AWI Atmospheric Physics by Sabine Scholle under the supervision of Felix Pithan. 

# Project Outline
We use the original 6-field AtmoRep model (wc5e2i3t) which was trained on wind velocity (or vorticity and divergence), vertical velocity, temperature, specific humidity and total precipitation. We then fine-tune it with ERA5 t2m data (new, unseen field) using BERT style masked token modelling. Finally, we apply this t2m-adapted model for continued fine-tuning using Akil Hoissain's corrected Arctic t2m data. In essence, introduce the model to ERA5 2m temperature, and then try nudge it to a better constrained Arctic environment for t2m bias-correction. 

A follow-up of this work, presented at EGU26 was an analysis of model performance as a function of increasing dataset sparsity. We simulated the effect of sparse observational Arctic t2m data, and analysed the effect on MSE token prediction of the model. 


# Starter README

## 1. Codebase structure in project ab1412

Below are the scripts specifically used in the context of the fine-tuning and sparsity evaluation experiments. For a description of the original codebase see https://github.com/clessig/atmorep. 


````
├── a270277                                       <- other project subdirectories
├── a270294
├── ...
└── atmorep/
    ├── atmorep/
    │     ├── core
                ├── nice_evaluation.py             <- evaluation script using N-ICE expedition data - this runs nice_evalution_single.py automatically
                └── train_corrected_era5.py       <- training script for fine-tuning on Akils's dataset
          ├── transformer             
          └── ...
    ├── data/                         <- helper and job scripts. Actual data at /work/ab1385/a270277
    │    ├── normalisation/           <- directory for data normalisations
    .    .
    ├── models
    │    ├── idiuw3ce3v               <- Directory containing model weights and config
    │    │       ├──  model_idiuw3ce3v.json     
    │    │       └──  AtmoRep_idiuw3ce3v.mod
    │    ├── id<model_id>
    .    .
    ├── plotting                       <- various scripts for plotting purposes
    .    .
    ├── results
         ├── idiuw3ce3v
         ...
    .    .
    └── universal_cleanup_runs.py      <- cleaning script for unwanted output files, logs, wandb runs, results, and models with associated job IDs to trash

````

## 2. Most important paths

Data: 
cf.file_path = '/work/ab1385/a270277/era5_y2010_2020_res25_corrected_t2m_copy.zarr' 

Models: 
- wc5e2i3t = original multi6 model from original codebase
- j2l0sz9j-19300146 = 6 field pretrained AtmoRep model, now fine-tuned on NEW era5 field, t2m, with cross attention between temp and t2m.
- iuw3ce3v-19413611 = 7 field Atmorep (with t2m above) fine-tuned on Akil's corrected t2m data 
- 6kjr71hd-23246602 = fine-tuning forecast model with corrected t2m




### 2.1 Download pre-trained models

Models can be downloaded from: https://datapub.fz-juelich.de/atmorep/trained-models.html

An example for downloading the pre-trained models is given here, in this case for the vorticity model.

`````
% atmorep/> mkdir models
% atmorep/> cd models
% atmorep/data/> wget https://datapub.fz-juelich.de/atmorep/models/model_id4nvwbetz.tar.gz
% atmorep/data/> tar xvzf model_id4nvwbetz.tar.gz
% atmorep/data/> ls id4nvwbetz
AtmoRep_id4nvwbetz.mod  model_id4nvwbetz.json
`````


### 2.2 Download model input data (ERA5)

The input data in the required structure can be downloaded from the [Jülich datapub](https://datapub.fz-juelich.de/atmorep/era5-data.html) server. Direct link to WebDAV [https://datapub.fz-juelich.de/atmorep/data/](https://datapub.fz-juelich.de/atmorep/data/). Alternatively, it can be directly downloaded from MARS using the following [script](https://www.atmorep.org/code/mars_era5_download.py).

#### Download a subset of files

All data files (fields and normalizations) should be downloaded into the ``data`` directory. Un-taring the files will generate the correct folder structure. For example (we will use the vorticity example also below to run the first model so it is recommended to download it as a first step):
`````
% atmorep/> mkdir data
% atmorep/> cd data
% atmorep/data/> wget https://datapub.fz-juelich.de/atmorep/data/vorticity/ml137/era5_vorticity_y2021_ml137.tar
% atmorep/data/> tar xvf era5_vorticity_y2021_ml137.tar
% atmorep/data/> ls -lah vorticity/ml137/
total 18G
era5_vorticity_y2021_m01_ml137.grib
era5_vorticity_y2021_m02_ml137.grib
...
era5_vorticity_y2021_m12_ml137.grib
`````
For efficiency reasons, AtmoRep takes monthly ERA5 data as input. Therefore, each tar file contains 12 GRIB files of about 1.5 GBytes each.

Coefficients for data normalization per field and level can be downloaded here: https://datapub.fz-juelich.de/atmorep/data/normalization/. They should also be located in the ```data``` directory:
`````
% atmorep/data/> wget https://datapub.fz-juelich.de/atmorep/data/normalization/normalization_vorticity_ml137.tar.gz
% atmorep/data/> tar xvzf normalization_vorticity_ml137.tar.gz
`````

## 3. Install python packages

Create a python environment, e.g.

`````
% atmorep/> python3 -m venv pyenv
`````

and activate the environment:

`````
% atmorep/> source pyenv/bin/activate
`````
conda is also possible, no environment is strictly required although we would recommend it. Please make sure to use a recent python version (we tested with python3.10).
Then install the AtmoRep package:
`````
% atmorep/>
% atmorep/> pip install -e .
`````

torch is currently not included (since it is often available or has particular dependencies, e.g. a specific Cuda version). In the simplest case, it can just be installed by:

`````
% atmorep/> pip install torch
`````
We require torch 2.x. (A container solution allows to run even on systems where torch 2.x is not available.)

## 4. Run model:
Pre-trained models can normally be run by:
`````
% atmorep/> python atmorep/core/evaluate.py
`````
You can easily adapt the configuration by selecting the corresponding _model_id_ in ``evaluate.py`` (see below). It defaults to the single-field configuration of vorticity, of which we have downloaded the data above.

Depending on your compute hardware, you might also have to run the computations by submitting the job using a batch system or allocate a compute node in interactive mode (if an interactive seesion is possible, then this is recommended). If you run an interactive session you will likely need to use the following:
`````
%  atmorep/> export CUDA_VISIBLE_DEVICES=0,1,2,3
%  atmorep/> MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
`````

The default evaluation mode is currently global forecast. The output will be (similar to) this:
````
devices : ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
Wandb run: atmorep-ztvyw7k6-8932958
Running Evaluate.evaluate with mode = global_forecast
Loaded AtmoRep id=4nvwbetz, ignoring/missing 2 elements.
Loaded model id = 4nvwbetz at epoch = -2.
Number of batches per global forecast: 14
INFO:: data stats vorticity : 5.374998363549821e-05 / 0.9978392720222473
num_accs_per_task : 1
with_hvd : True
hvd_rank : 0

...

wandb_id : ztvyw7k6
dates : [[2021, 2, 10, 12]]
token_overlap : [0, 0]
forecast_num_tokens : 1
validation loss for strategy=forecast at epoch 0 : 0.12402566522359848
validation loss for vorticity : 0.12402566522359848
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:        val. loss forecast ▁
wandb: val., forecast, vorticity ▁
wandb: 
wandb: Run summary:
wandb:        val. loss forecast 0.12403
wandb: val., forecast, vorticity 0.12403
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /p/project/atmo-rep/lessig/atmorep/atmorep/lessig-cleanup/atmorep/wandb/offline-run-20231124_095428-ztvyw7k6
````
For the vorticity example above, we evaluate with ``global_forecast`` for a specific date and using only a single model level:
````
mode, options = 'global_forecast', { 'fields[0][2]' : [137],
                                     'dates' : [ [2021, 2, 10, 12] ],
                                     'token_overlap' : [0, 0],
                                     'forecast_num_tokens' : 1, 
                                     'attention' : False}
````
We perform a 3 hour forecast, since 1 token is 3 hours wide. Another mode is the BERT masked token model mode used for pre-training:
`````
mode, options = 'BERT', {'years_test' : [2021], 'fields[0][2]' : [123, 137]}
`````
Again, we chose some custom options by using two levels instead of the five ones that are default and were used during pre-training and by using 2021 as the test year (since we downloaded the data).

The generated model output (stored in ``./results/id{wandbid}``) for the ```global_forecast``` example can be post-processed into a spatial map with the [following code](https://www.atmorep.org/code/plot_forecast.py). The run_id at the top needs to be replaced by the wandb_id of your run, it can be read off from the console output. Results will be stored as ``example_0000{0,1,2}.png``. The code is also an as-simple-as-possible example with many parameters hard-coded, see our analysis code for a proper handling. 
