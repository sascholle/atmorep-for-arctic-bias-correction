def nice_evaluation(): 
    """
    compare the data at 
    
    /work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc 

    air_temperature_2m:
    Shape: (3623,)
    Dtype: float64
    Attributes: ['_FillValue', 'instrument', 'units', 'long_name', 'instrumentAccuracy']
    First 5 values: [-- -- -- -- --]
    Last 5 values: [272.1299833333333 272.1724666666667 272.27734 -- --]
    
    and 

    /work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_ T2M_nice.nc 

    T2M:
    Shape: (3471,)
    Dtype: float32
    Attributes: ['_FillValue', 'long_name', 'units', 'code', 'table', 'CDI_grid_type', 'CDI_grid_num_LPE', 'coordinates']
    First 5 values: [240.61499 241.05482 241.20256 241.22261 241.32832]
    Last 5 values: [273.65753 273.63446 273.59485 273.51938 273.5279 ]


    and output of model 

    """

