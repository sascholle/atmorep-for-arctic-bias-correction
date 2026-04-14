import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_scatter_plot():
    # Read Datasets 
    ERA5_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/E5sf121H_201501_201506_T2M_nice.nc")
    NICE_data = xr.open_dataset("/work/ab1385/a270164/2024-sebai/data/N-ICE_MetSebData_2015_olre.nc")
    
    # Select the same time range from N-ICE
    NICE_data = NICE_data.sel(time=ERA5_data.time)
    
    # Extract variables
    ERA5_T2M = ERA5_data['T2M']
    NICE_T2M = NICE_data['air_temperature_2m']
    
    # Drop NaNs together
    combined = xr.Dataset({'ERA5_T2M': ERA5_T2M, 'NICE_T2M': NICE_T2M}).dropna(dim='time')
    ERA5_T2M_clean = combined['ERA5_T2M']
    NICE_T2M_clean = combined['NICE_T2M']
    
    print(f"ERA5 T2M range: {ERA5_T2M_clean.values.min():.2f} - {ERA5_T2M_clean.values.max():.2f} K")
    print(f"NICE T2M range: {NICE_T2M_clean.values.min():.2f} - {NICE_T2M_clean.values.max():.2f} K")
    print(f"Number of matched points: {len(NICE_T2M_clean)}")
    
    # Load Akil's corrected T2M .nc file
    akil_nc_path = "/work/ab1385/a270164/2024-sebai/data/era_corrected/prediction_T2M_arctic_2015.nc"
    akil_data = xr.open_dataset(akil_nc_path)
    akil_T2M = akil_data['T2M']
    
    print(f"\nAkil T2M range: {float(akil_T2M.min()):.2f} - {float(akil_T2M.max()):.2f} K")
    
    # Match Akil data to NICE observations
    akil_matched = []
    for t in NICE_T2M_clean.time.values:
        nice_lat = float(NICE_T2M_clean['lat'].sel(time=t).values)
        nice_lon = float(NICE_T2M_clean['lon'].sel(time=t).values)
        
        try:
            # Use xarray's sel with method='nearest'
            akil_val = float(akil_T2M.sel(
                time=t,
                lat=nice_lat,
                lon=nice_lon,
                method='nearest'
            ).values)
            akil_matched.append(akil_val)
        except:
            akil_matched.append(np.nan)
    
    akil_matched = np.array(akil_matched)
    valid_mask = ~np.isnan(akil_matched)
    print(f"Akil matched points: {np.sum(valid_mask)} / {len(akil_matched)}")
    if np.any(valid_mask):
        print(f"Akil matched range: {akil_matched[valid_mask].min():.2f} - {akil_matched[valid_mask].max():.2f} K")
    
    # Load AtmoRep Round7 matched data
    atmorep_nc_path = '/work/ab1412/atmorep/results/atmorep_nice_matched_valuesROUND7.nc'
    atmorep_data = xr.open_dataset(atmorep_nc_path)
    atmorep_vals = atmorep_data['atmorep_value'].values
    atmorep_nice_vals = atmorep_data['nice_value'].values
    atmorep_times = atmorep_data['time'].values
    
    print(f"\nAtmoRep matched points: {len(atmorep_vals)}")
    print(f"AtmoRep range: {atmorep_vals.min():.2f} - {atmorep_vals.max():.2f} K")
    
    # Convert to numpy for plotting
    ERA5_vals = ERA5_T2M_clean.values
    NICE_vals = NICE_T2M_clean.values
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ERA5 vs NICE
    ax.scatter(NICE_vals, ERA5_vals, 
               c='blue', alpha=0.6, s=50, label='ERA5 vs NICE', edgecolors='navy')
    
    # Plot Akil vs NICE (only valid points)
    if np.any(valid_mask):
        ax.scatter(NICE_vals[valid_mask], akil_matched[valid_mask],
                   c='green', alpha=0.6, s=50, label="Akil's corrected T2M vs NICE", 
                   edgecolors='darkgreen', marker='s')
    
    # Plot AtmoRep vs NICE
    ax.scatter(atmorep_nice_vals, atmorep_vals,
               c='red', alpha=0.6, s=50, label='AtmoRep vs NICE',
               edgecolors='darkred', marker='^')
    
    # Add 1:1 line
    if np.any(valid_mask):
        all_vals = np.concatenate([NICE_vals, ERA5_vals, akil_matched[valid_mask]])
    else:
        all_vals = np.concatenate([NICE_vals, ERA5_vals])
    min_val = np.nanmin(all_vals) - 2
    max_val = np.nanmax(all_vals) + 2
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')
    
    # Calculate and display RMSE
    rmse_era5 = np.sqrt(np.mean((ERA5_vals - NICE_vals) ** 2))
    bias_era5 = np.mean(ERA5_vals - NICE_vals)
    
    stats_text = f'ERA5 vs NICE:\n  RMSE = {rmse_era5:.2f} K\n  Bias = {bias_era5:.2f} K\n  N = {len(NICE_vals)}'
    
    if np.any(valid_mask):
        rmse_akil = np.sqrt(np.mean((akil_matched[valid_mask] - NICE_vals[valid_mask]) ** 2))
        bias_akil = np.mean(akil_matched[valid_mask] - NICE_vals[valid_mask])
        stats_text += f"\n\nAkil vs NICE:\n  RMSE = {rmse_akil:.2f} K\n  Bias = {bias_akil:.2f} K\n  N = {np.sum(valid_mask)}"
   
    # AtmoRep stats
    rmse_atmorep = np.sqrt(np.mean((atmorep_vals - atmorep_nice_vals) ** 2))
    bias_atmorep = np.mean(atmorep_vals - atmorep_nice_vals)
    stats_text += f"\n\nAtmoRep vs NICE:\n  RMSE = {rmse_atmorep:.2f} K\n  Bias = {bias_atmorep:.2f} K\n  N = {len(atmorep_vals)}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('NICE T2M (K)', fontsize=14)
    ax.set_ylabel('Model/ERA5 T2M (K)', fontsize=14)
    ax.set_title('Scatter: 2015 ERA5, Akil & AtmoRep corrected T2M vs NICE\n(N-ICE ship observations)', fontsize=14)
    
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/work/ab1412/atmorep/plotting/scatter_era5_akil_atmorep_vs_niceROUND7first_timestep.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to /work/ab1412/atmorep/plotting/scatter_era5_akil_atmorep_vs_niceROUND7first_timestep.png")
    plt.close()

if __name__ == "__main__":
    create_scatter_plot()