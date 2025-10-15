import zarr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os

def analyze_t2m_seasonality(zarr_path):
    """Analyze seasonal patterns in corrected T2M data"""
    print(f"=== T2M Seasonality Analysis ===")
    print(f"Opening: {zarr_path}")
    
    store = zarr.open(zarr_path, mode='r')
    data = store['data_sfc']  # Shape: (105192, 1, 721, 1440)
    #print time file type 
    times = store['time'][:]
    lats = store['lats'][:]
    lons = store['lons'][:]
    
    print(f"Data sfc shape: {data.shape}")
    print(f"Time range: {times[0]} to {times[-1]}, type: {type(times)}")
    print(f"Lat range: {lats[0]:.2f} to {lats[-1]:.2f}")  
    print(f"Lon range: {lons[0]:.2f} to {lons[-1]:.2f}")

    if isinstance(times[0], np.datetime64):
        print("Detected numpy datetime64 format")
        # Convert to pandas datetime for easier manipulation
        times_pd = pd.to_datetime(times)
        months = times_pd.month.values
        years = times_pd.year.values
        
        # Sample every 24 hours (daily) to reduce computation
        daily_step = 24  # Assuming hourly data
        time_sample = slice(0, None, daily_step)
    else:
        # Fallback for numeric time values
        print("Warning: Time format not recognized, using approximate conversion")
        # Assume hours since epoch, convert to months
        hours_per_month = 24 * 30.44  # Average month length
        months = ((times / hours_per_month) % 12).astype(int) + 1
        time_sample = slice(0, None, 24)
     
    arctic_lats = slice(0, 71)  # Northern part (assuming 0 is north pole)
    
    print("Calculating monthly statistics...")
    sample_data = data[time_sample, 0, arctic_lats, ::10]  # Sample every 10th longitude
    sample_months = months[time_sample] 

    # Group by month (approximate)
    hours_per_month = 24 * 30  # Approximate
    
    monthly_stats = []
    for month in range(1, 13):
        mask = sample_months == month
        if mask.sum() > 0:
            month_data = sample_data[mask]
            stats = {
                'month': month,
                'mean_temp': np.nanmean(month_data),
                'min_temp': np.nanmin(month_data),
                'max_temp': np.nanmax(month_data),
                'std_temp': np.nanstd(month_data),
                'count': mask.sum()
            }
            monthly_stats.append(stats)
    
    df_monthly = pd.DataFrame(monthly_stats)
    print("\nMonthly Temperature Statistics (Arctic Region):")
    print(df_monthly.round(2))
    
    # Plot seasonality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Monthly means
    ax1.plot(df_monthly['month'], df_monthly['mean_temp'], 'bo-', linewidth=2)
    ax1.fill_between(df_monthly['month'], 
                     df_monthly['mean_temp'] - df_monthly['std_temp'],
                     df_monthly['mean_temp'] + df_monthly['std_temp'], 
                     alpha=0.3)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Arctic T2M Monthly Mean Seasonal Pattern')
    ax1.grid(True)
    ax1.set_xticks(range(1, 13))
    
    # Min/Max range
    ax2.plot(df_monthly['month'], df_monthly['max_temp'], 'r-', label='Max', linewidth=2)
    ax2.plot(df_monthly['month'], df_monthly['min_temp'], 'b-', label='Min', linewidth=2)
    ax2.fill_between(df_monthly['month'], df_monthly['min_temp'], df_monthly['max_temp'], alpha=0.2)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Arctic T2M Monthly Range')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(range(1, 13))
    
    plt.tight_layout()
    plt.savefig('/work/ab1412/atmorep/plotting/t2m_seasonality/t2m_seasonality.png', dpi=150, bbox_inches='tight')
    print(f"Seasonality plot saved to: /work/ab1412/atmorep/plotting/t2m_seasonality/t2m_seasonality.png")

    return df_monthly

def check_normalization_consistency():
    """Check if norm_sfc values match actual monthly data statistics at individual grid points"""
    print(f"\n=== Normalization Consistency Check (Grid Point Level) ===")
    
    # Load the actual T2M data
    t2m_path = "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr"
    t2m_store = zarr.open(t2m_path, mode='r')
    t2m_data = t2m_store['data_sfc']  # Shape: (105192, 1, 721, 1440)
    times = t2m_store['time'][:]
    
    # Load the normalization values
    norm_path = "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr"
    norm_store = zarr.open(norm_path, mode='r')
    
    if 'normalization/norm_sfc' not in norm_store:
        print("ERROR: normalization/norm_sfc not found in norm store!")
        return
    
    norm_data = norm_store['normalization/norm_sfc']  # Shape: (144, 2, 3, 721, 1440)
    print(f"T2M data shape: {t2m_data.shape}")
    print(f"Norm data shape: {norm_data.shape}")
    
    # Convert times to months
    if isinstance(times[0], np.datetime64):
        times_pd = pd.to_datetime(times)
        years = times_pd.year.values
        months = times_pd.month.values
        
        # Create month indices (0-143 for 144 months total)
        start_year = years[0]
        month_indices = (years - start_year) * 12 + (months - 1)
        
    else:
        print("ERROR: Cannot handle non-datetime time format")
        return
    
    # Use last field (assumed to be corrected_t2m)
    t2m_field_idx = -1
    print(f"Using field index: {t2m_field_idx} (assuming corrected_t2m)")
    
    # Extract normalization values for corrected_t2m
    norm_means = norm_data[:, 0, t2m_field_idx, :, :]  # (144, 721, 1440)
    norm_stds = norm_data[:, 1, t2m_field_idx, :, :]   # (144, 721, 1440)
    
    # Select a few Arctic grid points to test
    def sample_arctic_points(num_points=100):
        # Arctic latitudes: 0 to 71 (inclusive)
        lat_indices = np.random.randint(50, 72, num_points)
        lon_indices = np.linspace(0, 1439, num_points, dtype=int)
        points = []
        for lat, lon in zip(lat_indices, lon_indices):
            points.append((lat, lon))
            if len(points) >= num_points:
                break
        return points
    test_points = sample_arctic_points(num_points=100)
    
    print(f"Testing {len(test_points)} Arctic grid points across first few months...")
    print(f"Grid points (lat_idx, lon_idx): {test_points}")
    
    grid_point_results = []
    
    # Test first 6 months
    for month_idx in range(min(6, norm_means.shape[0])):
        
        # Get time indices for this month
        month_mask = month_indices == month_idx
        month_time_indices = np.where(month_mask)[0]
        
        if len(month_time_indices) == 0:
            print(f"No data found for month {month_idx}")
            continue
            
        print(f"\nMonth {month_idx}: {len(month_time_indices)} time steps")
        
        for lat_idx, lon_idx in test_points:
            
            # Get normalization values for this grid point and month
            norm_mean = norm_means[month_idx, lat_idx, lon_idx]
            norm_std = norm_stds[month_idx, lat_idx, lon_idx]
            
            # Skip if normalization values are NaN
            if np.isnan(norm_mean) or np.isnan(norm_std):
                print(f"  Grid ({lat_idx:2d},{lon_idx:3d}): Norm values are NaN, skipping")
                continue
            
            # Extract actual data for this grid point and month
            # Load all time steps for this month at this grid point
            grid_point_data = []
            
            # Sample every 10th time step to reduce computation
            sample_indices = month_time_indices  
            
            for time_idx in sample_indices:
                value = t2m_data[time_idx, 0, lat_idx, lon_idx]
                grid_point_data.append(value)
            
            grid_point_data = np.array(grid_point_data)
            
            # Calculate actual statistics for this grid point and month
            actual_mean = np.nanmean(grid_point_data)
            actual_std = np.nanstd(grid_point_data)
            
            # Calculate differences
            mean_diff = abs(actual_mean - norm_mean)
            std_diff = abs(actual_std - norm_std)
            
            result = {
                'month_idx': month_idx,
                'lat_idx': lat_idx,
                'lon_idx': lon_idx,
                'actual_mean': actual_mean,
                'norm_mean': norm_mean,
                'actual_std': actual_std,
                'norm_std': norm_std,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'n_samples': len(grid_point_data)
            }
            grid_point_results.append(result)
            
            # Print individual result
            print(f"  Grid ({lat_idx:2d},{lon_idx:3d}): "
                  f"Actual(μ={actual_mean:.1f}K,σ={actual_std:.1f}K) "
                  f"Norm(μ={norm_mean:.1f}K,σ={norm_std:.1f}K) "
                  f"Diff(Δμ={mean_diff:.1f}K,Δσ={std_diff:.1f}K) "
                  f"N={len(grid_point_data)}")
    
    # Overall assessment
    if grid_point_results:
        df_results = pd.DataFrame(grid_point_results)
        
        print(f"\n=== Grid Point Comparison Summary ===")
        print(f"Total comparisons: {len(df_results)}")
        print(f"Average mean difference: {df_results['mean_diff'].mean():.2f} K")
        print(f"Average std difference:  {df_results['std_diff'].mean():.2f} K")
        print(f"Max mean difference:     {df_results['mean_diff'].max():.2f} K")
        print(f"Max std difference:      {df_results['std_diff'].max():.2f} K")
        
        # Consistency thresholds (tighter since we're comparing grid points directly)
        mean_threshold = 1.0  # 1K tolerance for grid point means
        std_threshold = 0.5   # 0.5K tolerance for grid point stds
        
        mean_consistent = (df_results['mean_diff'] < mean_threshold).sum()
        std_consistent = (df_results['std_diff'] < std_threshold).sum()
        total_comparisons = len(df_results)
        
        print(f"\nConsistency Check:")
        print(f"Mean consistency: {mean_consistent}/{total_comparisons} ({100*mean_consistent/total_comparisons:.0f}%)")
        print(f"Std consistency:  {std_consistent}/{total_comparisons} ({100*std_consistent/total_comparisons:.0f}%)")
        
        # Show worst cases
        worst_mean = df_results.loc[df_results['mean_diff'].idxmax()]
        worst_std = df_results.loc[df_results['std_diff'].idxmax()]
        
        print(f"\nWorst mean difference:")
        print(f"  Month {worst_mean['month_idx']}, Grid ({worst_mean['lat_idx']},{worst_mean['lon_idx']}): {worst_mean['mean_diff']:.2f}K")
        print(f"Worst std difference:")
        print(f"  Month {worst_std['month_idx']}, Grid ({worst_std['lat_idx']},{worst_std['lon_idx']}): {worst_std['std_diff']:.2f}K")
        
        overall_consistent = (mean_consistent/total_comparisons > 0.8) and (std_consistent/total_comparisons > 0.8)
        
        if overall_consistent:
            print("🎉 Grid-point normalization values are consistent with data!")
        else:
            print("⚠️  Grid-point normalization values may not match the data well.")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean comparison scatter
        ax1.scatter(df_results['actual_mean'], df_results['norm_mean'], alpha=0.7)
        min_val = min(df_results['actual_mean'].min(), df_results['norm_mean'].min())
        max_val = max(df_results['actual_mean'].max(), df_results['norm_mean'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match')
        ax1.set_xlabel('Actual Grid Point Mean (K)')
        ax1.set_ylabel('Norm Grid Point Mean (K)')
        ax1.set_title('Grid Point Mean Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Std comparison scatter  
        ax2.scatter(df_results['actual_std'], df_results['norm_std'], alpha=0.7)
        min_val = min(df_results['actual_std'].min(), df_results['norm_std'].min())
        max_val = max(df_results['actual_std'].max(), df_results['norm_std'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match')
        ax2.set_xlabel('Actual Grid Point Std (K)')
        ax2.set_ylabel('Norm Grid Point Std (K)')
        ax2.set_title('Grid Point Std Comparison')
        ax2.legend()
        ax2.grid(True)
        
        # Difference histograms
        ax3.hist(df_results['mean_diff'], bins=20, alpha=0.7, color='blue')
        ax3.axvline(mean_threshold, color='red', linestyle='--', label=f'Threshold ({mean_threshold}K)')
        ax3.set_xlabel('Mean Difference (K)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Mean Differences')
        ax3.legend()
        ax3.grid(True)
        
        ax4.hist(df_results['std_diff'], bins=20, alpha=0.7, color='orange')
        ax4.axvline(std_threshold, color='red', linestyle='--', label=f'Threshold ({std_threshold}K)')
        ax4.set_xlabel('Std Difference (K)')
        ax4.set_ylabel('Frequency') 
        ax4.set_title('Distribution of Std Differences')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        os.makedirs('/work/ab1412/atmorep/plotting', exist_ok=True)
        plt.savefig('/work/ab1412/atmorep/plotting/grid_point_norm_check_all_timesteps.png', dpi=150, bbox_inches='tight')
        print(f"Grid point comparison plot saved to: /work/ab1412/atmorep/plotting/grid_point_norm_check.png")
        plt.close()
        
        return df_results
    
    else:
        print("ERROR: No valid grid point comparisons could be made")
        return None
    
def quick_data_overview():
    """Quick overview of both datasets"""
    print(f"\n=== Quick Data Overview ===")
    
    paths = {
        'T2M Data': "/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr",
        'Norm Data': "/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr"
    }
    
    for name, path in paths.items():
        print(f"\n{name} ({path}):")
        try:
            store = zarr.open(path, mode='r')
            print(f"  Arrays: {list(store.array_keys())}")
            
            if name == 'T2M Data':
                data_sfc = store['data_sfc']
                print(f"  Data sfc shape: {data_sfc.shape}")
                print(f"  Data dtype: {data_sfc.dtype}")
                sample = data_sfc[0, 0, 0:71, 0:71]  # Small sample
                print(f"  Sample mean: {np.nanmean(sample):.2f}")
                print(f"  Sample range: {np.nanmin(sample):.2f} to {np.nanmax(sample):.2f}")
            elif name == 'Norm Data':
                norm_sfc = store['normalization/norm_sfc']
                print(f"  Norm sfc shape: {norm_sfc.shape}")
                print(f"  Norm sfc dtype: {norm_sfc.dtype}")
                sample_mean = norm_sfc[:, 0, -1, 0:105, 0:105]  # Last field (assumed corrected_t2m)
                sample_std = norm_sfc[:, 1, -1, 0:105, 0:105]
                print(f" corrected_t2m norm sfc sample mean: {np.nanmean(sample_mean):.2f}")
                print(f" corrected_t2m norm sfc sample range: {np.nanmin(sample_mean):.2f} to {np.nanmax(sample_mean):.2f}")
                print(f" corrected_t2m norm sfc sample std: {np.nanmean(sample_std):.2f}")
                print(f" corrected_t2m norm sfc sample range: {np.nanmin(sample_std):.2f} to {np.nanmax(sample_std):.2f}")

        except Exception as e:
            print(f"  Error: {e}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs('/work/ab1412/atmorep/plotting', exist_ok=True)
    
    print("Starting Corrected T2M Analysis...")
    print("=" * 60)
    
    # Quick overview
    quick_data_overview()
    
    # # Analyze seasonality
    # try:
    #     monthly_stats = analyze_t2m_seasonality("/scratch/a/a270277/atmorep/data_t2m_Akil_padded.zarr")
    # except Exception as e:
    #     print(f"Error in seasonality analysis: {e}")
    
    # Check normalization consistency
    try:
        norm_check = check_normalization_consistency()
    except Exception as e:
        print(f"Error in normalization check: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()