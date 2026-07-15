# --- Configuration ---
BASE_DIR = '/scratch/wofs/matthew.ammon/nature_runs/fcst/20250314'
BASELINE_NAME = 'SfcRad_Test'  # Now set as the baseline
NR_DIR = '/work2/wof/realtime/nature_run/20250314/nr_ens0'
NR_PREFIX = 'wrfnr_d01'

NUM_MEMBERS = 18
PERCENTILE_THRESH = 95.0
NEIGHBORHOOD_RADIUS = 3
SMOOTHING_SIGMA = 2

exps = ['SfcRad_Test','SfcRad_UAS', 'SfcRad_20UAS', 'SfcRad_20UAS_XLV',
        'SfcRad_20UAS_2XLV', 'SfcRad_20UAS_smHlgV']

############################################


import os
import glob
import math
import concurrent.futures
from functools import partial
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, maximum_filter
import pyresample

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# --- Global Cartopy Shapefile Setup ---
projection = ccrs.PlateCarree()

try:
    reader_counties = shpreader.Reader('/home/matthew.ammon/SHAPEFILES/US_COUNTIES_5m/countyl010g.shp')
    COUNTIES = cfeature.ShapelyFeature(list(reader_counties.geometries()), projection)

    reader_states = shpreader.Reader('/home/matthew.ammon/SHAPEFILES/US_STATES_5m/cb_2018_us_state_5m.shp')
    STATES = cfeature.ShapelyFeature(list(reader_states.geometries()), projection)
except Exception as e:
    print(f"Warning: Could not load shapefiles. Ensure paths are correct. Error: {e}")
    COUNTIES = cfeature.BORDERS
    STATES = cfeature.STATES


# --- Helper Functions ---

def subsection_idxs(area, lat, lon):
    lon_min, lon_max, lat_min, lat_max = area
    ilat = np.where((lat[:,0] >= lat_min) & (lat[:,0] <= lat_max))[0]
    ilon = np.where((lon[0,:] >= lon_min) & (lon[0,:] <= lon_max))[0]
    return ilat.min(), ilat.max(), ilon.min(), ilon.max()

def resample(data, old_lat, old_lon, new_lat, new_lon):
    orig_def = pyresample.geometry.SwathDefinition(lons=old_lon, lats=old_lat)
    targ_def = pyresample.geometry.SwathDefinition(lons=new_lon, lats=new_lat)
    resampled_variable = pyresample.kd_tree.resample_nearest(
        orig_def, data, targ_def, radius_of_influence=5000, fill_value=0
    )
    return resampled_variable

def create_circular_footprint(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask

def get_ens_nmep(ens_dir, time_str, ens_threshold, footprint):
    """Calculates smoothed NMEP for a given ensemble directory at a specific time."""
    member_grids = []

    for member_num in range(1, NUM_MEMBERS + 1):
        mem_str = f"ENS_MEM_{member_num:02d}"
        mem_file = f"{ens_dir}/{mem_str}/wrfout_d01_{time_str}"

        if not os.path.exists(mem_file):
            continue

        with xr.open_dataset(mem_file) as ds_mem:
            uh_mem = ds_mem.UP_HELI_MAX.data[0,:,:]

        binary_uh = (uh_mem >= ens_threshold).astype(float)
        neighborhood_uh = maximum_filter(binary_uh, footprint=footprint)
        member_grids.append(neighborhood_uh)

    if not member_grids:
        return None

    ensemble_stack = np.stack(member_grids, axis=0)
    nep_array = np.mean(ensemble_stack, axis=0)
    smoothed_nep = gaussian_filter(nep_array, sigma=SMOOTHING_SIGMA, mode='nearest')

    return smoothed_nep

def process_time_step(time_str, area, lat_ens, lon_ens, lat_nr_trim, lon_nr_trim, lat1, lat2, lon1, lon2, base_dir, test_dirs):
    """Processes a time step: calculates NMEP for all exps, computes BSS, and plots dynamically."""
    dt_obj = datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%S")
    nr_file = f"{NR_DIR}/{NR_PREFIX}_{time_str.replace(':','_')}"

    if not os.path.exists(nr_file):
        print(f"Warning: NR file missing for {time_str}. Skipping.")
        return None

    # 1. Process Nature Run (With HH:00 hourly accumulation reset logic)
    with xr.open_dataset(nr_file) as ds_nr:
        uh_nr_curr = ds_nr.UP_HELI_MAX.data[0,:,:]

    if dt_obj.minute == 15:
        uh_nr_full = uh_nr_curr
    else:
        dt_prev = dt_obj - timedelta(minutes=15)
        time_str_prev = dt_prev.strftime("%Y-%m-%d_%H:%M:%S")
        nr_file_prev = f"{NR_DIR}/{NR_PREFIX}_{time_str_prev.replace(':','_')}"

        if os.path.exists(nr_file_prev):
            with xr.open_dataset(nr_file_prev) as ds_nr_prev:
                uh_nr_prev = ds_nr_prev.UP_HELI_MAX.data[0,:,:]
            uh_nr_full = np.maximum(uh_nr_curr - uh_nr_prev, 0)
        else:
            uh_nr_full = uh_nr_curr

    uh_nr_trim = uh_nr_full[lat1:lat2+1, lon1:lon2+1]
    uh_nr_trim = gaussian_filter(uh_nr_trim, sigma=3, mode='nearest')

    uh_threshold = np.percentile(uh_nr_trim[uh_nr_trim>0.1], PERCENTILE_THRESH)
    ens_threshold = uh_threshold / 5.04
    truth_mask = (uh_nr_trim >= uh_threshold).astype(float)

    footprint = create_circular_footprint(NEIGHBORHOOD_RADIUS)

    # 2. Process All Experiments
    bs_vals = {}
    resampled_grids = {}

    # Process Baseline
    nmep_base = get_ens_nmep(base_dir, time_str, ens_threshold, footprint)
    if nmep_base is None:
        return None
    resampled_base = resample(nmep_base, lat_ens, lon_ens, lat_nr_trim, lon_nr_trim)
    bs_base = np.nanmean((resampled_base - truth_mask)**2)
    bs_vals[BASELINE_NAME] = bs_base
    resampled_grids[BASELINE_NAME] = resampled_base

    # Process Tests
    bss_dict = {}
    for test_name, test_dir in test_dirs.items():
        nmep_test = get_ens_nmep(test_dir, time_str, ens_threshold, footprint)
        if nmep_test is None:
            continue
        resampled_t = resample(nmep_test, lat_ens, lon_ens, lat_nr_trim, lon_nr_trim)
        bs_t = np.nanmean((resampled_t - truth_mask)**2)

        bs_vals[test_name] = bs_t
        resampled_grids[test_name] = resampled_t
        bss_dict[test_name] = 1.0 - (bs_t / bs_base) if bs_base > 0 else np.nan

    # Identify Best and Worst Brier Scores for color-coding
    valid_bs = {k: v for k, v in bs_vals.items() if not np.isnan(v)}
    best_exp = min(valid_bs, key=valid_bs.get) if valid_bs else None
    worst_exp = max(valid_bs, key=valid_bs.get) if valid_bs else None

    # 3. Dynamic Multi-Panel Spatial Plotting
    n_panels = len(resampled_grids)
    ncols = min(n_panels, 3)
    nrows = math.ceil(n_panels / ncols)

    proj = ccrs.LambertConformal(
        central_longitude=np.nanmean(lon_nr_trim),
        central_latitude=np.nanmean(lat_nr_trim)
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), subplot_kw={'projection': proj})
    if n_panels == 1: axes = [axes]
    else: axes = axes.flatten()

    levels = np.arange(0.1, 1.05, 0.05)
    fill_plot = None

    for idx, (exp_name, grid) in enumerate(resampled_grids.items()):
        ax = axes[idx]
        ax.add_feature(COUNTIES, facecolor='none', edgecolor='black', alpha=0.1, zorder=4)
        ax.add_feature(STATES, facecolor='none', edgecolor='black', linewidth=1, alpha=0.3, zorder=4)

        fill_plot = ax.contourf(lon_nr_trim, lat_nr_trim, grid, levels=levels, cmap='afmhot_r', alpha=0.8, transform=ccrs.PlateCarree(), extend='max', zorder=3)
        ax.contour(lon_nr_trim, lat_nr_trim, uh_nr_trim, levels=[uh_threshold], colors='k', linewidths=1, transform=ccrs.PlateCarree(), zorder=5)
        ax.set_extent(area)

        # Color Code BS Value
        bs_val = bs_vals[exp_name]
        if exp_name == best_exp:
            bs_color = 'green'
        elif exp_name == worst_exp:
            bs_color = 'red'
        else:
            bs_color = 'black'

        ax.set_title(f"{exp_name}", fontsize=13, fontweight='bold', pad=22)
        ax.text(0.5, 1.02, f"BS: {bs_val:.4f}", transform=ax.transAxes, ha='center', va='bottom', color=bs_color, fontsize=12, fontweight='bold')

    # Hide any unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    # Master Colorbar
    cbar_ax = fig.add_axes([0.15, 0.05 if nrows == 1 else 0.02, 0.7, 0.03])
    cbar = fig.colorbar(fill_plot, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f"NMEP (UH > {ens_threshold:.1f})")

    fig.suptitle(f"Valid: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')} UTC | Nature Run Threshold: {PERCENTILE_THRESH}th %ile", fontsize=15, fontweight='bold', y=1.05 if nrows == 1 else 1.02)

    os.makedirs("NEP/Figs/temp", exist_ok=True)
    spatial_plot_filename = f"NEP/Figs/temp/Comparison_Spatial_{dt_obj.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(spatial_plot_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return (dt_obj, bss_dict)

# --- Main Script ---

def main(exps):
    #print("Auto-detecting experiments...")
    #all_exp_paths = glob.glob(f"{BASE_DIR}/*/2300")
    all_exp_paths = []

    for e in exps:
        if e == 'SfcRad_UAS':
            all_exp_paths.append('/work/joshua.gebauer/Big_NatureRun/fcst/20250314/SfcRad_UAS_all/2300')
        elif e == 'SfcRad':
            all_exp_paths.append('/work/joshua.gebauer/Big_NatureRun/fcst/20250314/SfcRad/2300')
        else:
            all_exp_paths.append(f'{BASE_DIR}/{e}/2300')

    # Isolate Baseline and Test Directories
    experiments = {}
    for p in all_exp_paths:
        name = p.split('/')[-2]
        experiments[name] = p

    if BASELINE_NAME not in experiments:
        raise ValueError(f"Baseline '{BASELINE_NAME}' not found in {BASE_DIR}")

    baseline_dir = experiments.pop(BASELINE_NAME)
    test_dirs = experiments # Remaining are tests
    test_names = list(test_dirs.keys())

    print(f"Baseline: {BASELINE_NAME}")
    print(f"Test Experiments detected: {test_names}")

    # Generate reference files off the Baseline directory
    all_ens_files = sorted(glob.glob(f'{baseline_dir}/ENS_MEM_01/wrfout_d01*'))
    if not all_ens_files:
        raise ValueError(f"No baseline ensemble files found in {baseline_dir}.")

    time_strs = []
    for f in all_ens_files:
        t_str = os.path.basename(f).split('_d01_')[-1]
        dt = datetime.strptime(t_str, "%Y-%m-%d_%H:%M:%S")
        if dt.minute % 15 == 0:
            time_strs.append(t_str)

    print(f"Filtered to {len(time_strs)} time steps at 15-minute intervals.")

    with xr.open_dataset(all_ens_files[0]) as ds_ens:
        lat_ens = ds_ens.XLAT.data[0,:,:]
        lon_ens = ds_ens.XLONG.data[0,:,:]

    area = [np.nanmin(lon_ens), np.nanmax(lon_ens), np.nanmin(lat_ens), np.nanmax(lat_ens)]
    first_nr_file = f"{NR_DIR}/{NR_PREFIX}_{time_strs[0].replace(':','_')}"

    with xr.open_dataset(first_nr_file) as ds_nr_init:
        lat_nr_full = ds_nr_init.XLAT.data[0,:,:]
        lon_nr_full = ds_nr_init.XLONG.data[0,:,:]

    lat1, lat2, lon1, lon2 = subsection_idxs(area, lat_nr_full, lon_nr_full)
    lat_nr_trim = lat_nr_full[lat1:lat2+1, lon1:lon2+1]
    lon_nr_trim = lon_nr_full[lat1:lat2+1, lon1:lon2+1]

    print("Executing computations and generating comparison plots in parallel...")
    worker_func = partial(
        process_time_step,
        area=area, lat_ens=lat_ens, lon_ens=lon_ens,
        lat_nr_trim=lat_nr_trim, lon_nr_trim=lon_nr_trim,
        lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2,
        base_dir=baseline_dir, test_dirs=test_dirs
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_MEMBERS) as executor:
        results = list(executor.map(worker_func, time_strs))

    valid_results = [res for res in results if res is not None]
    valid_results.sort(key=lambda x: x[0])

    if not valid_results:
        raise ValueError("No valid scores generated.")

    times = [res[0] for res in valid_results]
    bss_dicts = [res[1] for res in valid_results]

    print("Generating Master Brier Skill Score Time Series Plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--') # Zero skill line

    colors = plt.cm.tab10.colors

    # Plot a line for each test experiment
    for i, test_name in enumerate(test_names):
        skill_scores = [d.get(test_name, np.nan) for d in bss_dicts]
        ax.plot(times, skill_scores, marker='o', linestyle='-', color=colors[i % 10], linewidth=2, markersize=7, label=test_name)

    ax.set_title(f"15-Minute Time Series: Brier Skill Score vs '{BASELINE_NAME}'\n({PERCENTILE_THRESH}th Percentile NR Threshold)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (UTC)", fontsize=12)
    ax.set_ylabel("Brier Skill Score (BSS)", fontsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    fig.autofmt_xdate(rotation=45)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=11)

    output_filename = "NMEP_BSS_Multiexp_Timeseries.png"
    plt.tight_layout()
    plt.show()

main(exps)
