import os
import glob
import concurrent.futures
from functools import partial

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, maximum_filter

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from utils import *
from config import *

projection = ccrs.PlateCarree()
reader = shpreader.Reader('/home/matthew.ammon/SHAPEFILES/US_COUNTIES_5m/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, projection)

reader = shpreader.Reader('/home/matthew.ammon/SHAPEFILES/US_STATES_5m/cb_2018_us_state_5m.shp')
states = list(reader.geometries())
STATES = cfeature.ShapelyFeature(states, projection)

# --- Main Script ---

def main(exp, swt_var, ENS_DIR, NR_DIR,
         NUM_MEMBERS, PERCENTILE_THRESH,
         NEIGHBORHOOD_RADIUS, SMOOTHING_SIGMA):

    # 1. Setup Base Ensemble Information
    ens1_files = sorted(glob.glob(f'{ENS_DIR}/ENS_MEM_01/wrfout_d01*'))

    # Extract times from ensemble files (Format: wrfout_d01_YYYY-MM-DD_HH:MM:SS)
    times = [os.path.basename(f).split('_d01_')[-1] for f in ens1_files]
    nr_files_matching = [f"{NR_DIR}/{NR_PREFIX}_{t.replace(':','_')}" for t in times]

    # Get base lat/lon bounds from the ensemble
    with xr.open_dataset(ens1_files[0]) as ds_ens:
        lat_ens = ds_ens.XLAT.data[0,:,:]
        lon_ens = ds_ens.XLONG.data[0,:,:]

    area = [np.nanmin(lon_ens), np.nanmax(lon_ens), np.nanmin(lat_ens), np.nanmax(lat_ens)]

    # 2. Process the Nature Run
    print("Accumulating Nature Run data...")
    for i, f in enumerate(nr_files_matching):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Nature run file missing for time step: {f}")

        with xr.open_dataset(f) as ds_nr:
            if i == 0:
                lat_nr_full = ds_nr.XLAT.data[0,:,:]
                lon_nr_full = ds_nr.XLONG.data[0,:,:]
                uh_nr_full = ds_nr[ds_swt_var[swt_var]].data[0,:,:]
            else:
                uh_nr_full = uh_nr_full + ds_nr[ds_swt_var[swt_var]].data[0,:,:]

    # Trim the Nature Run to the Ensemble domain using the bounds
    lat1, lat2, lon1, lon2 = subsection_idxs(area, lat_nr_full, lon_nr_full)

    # Slicing the full nature run arrays (Note: +1 to make slice inclusive of bounds)
    uh_nr_trim = uh_nr_full[lat1:lat2+1, lon1:lon2+1]
    uh_nr_trim = gaussian_filter(uh_nr_trim, sigma=3, mode='nearest')
    lat_nr_trim = lat_nr_full[lat1:lat2+1, lon1:lon2+1]
    lon_nr_trim = lon_nr_full[lat1:lat2+1, lon1:lon2+1]

    # Calculate the percentile threshold based on all grid points in the trimmed NR domain
    uh_threshold = np.percentile(uh_nr_trim[uh_nr_trim>0], PERCENTILE_THRESH)
    print(f"Calculated {PERCENTILE_THRESH}th percentile UH threshold: {uh_threshold:.2f}")

    # 3. Process the Ensemble Members (Parallelized)
    print("Processing Ensemble NMEP in parallel...")
    member_numbers = range(1, NUM_MEMBERS + 1)

    # Using functools.partial to pass the shared threshold & file list into our worker function
    worker_func = partial(process_member, threshold=uh_threshold,
                          ens_files_base=ens1_files, rad = NEIGHBORHOOD_RADIUS, swt_var = swt_var)

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_MEMBERS) as executor:
        results = list(executor.map(worker_func, member_numbers))

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        raise ValueError("No valid ensemble data processed.")

    # Stack array, get fractional probability, and smooth it
    ensemble_stack = np.stack(valid_results, axis=0)
    nep_array = np.mean(ensemble_stack, axis=0)
    smoothed_nep = gaussian_filter(nep_array, sigma=SMOOTHING_SIGMA, mode='nearest')

    # 4. Resample NMEP to the trimmed Nature Run grid
    print("Resampling NMEP to Nature Run grid...")
    resampled_nmep = resample(smoothed_nep, lat_ens, lon_ens, lat_nr_trim, lon_nr_trim)

    # --- Metrics Calculation ---
    print("Computing POD and FAR...")

    # Create a boolean mask indicating where the NR Truth exceeds the threshold
    truth_mask = uh_nr_trim >= uh_threshold

    # POD: Average NMEP across all gridpoints inside the threshold area
    # using np.nanmean to safely ignore any nan values that may exist from resampling
    pod = np.nanmean(resampled_nmep[truth_mask])

    # FAR: Average NMEP across all gridpoints outside the threshold area
    far = np.nanmean(resampled_nmep[~truth_mask]) #+ (np.nansum((resampled_nmep[~truth_mask]>0))/np.nansum(~truth_mask))

    print(f"Calculated POD: {pod:.4f}")
    print(f"Calculated FAR: {far:.4f}")

    # 5. Create the Figure
    print("Generating Figure...")
    fig = plt.figure(figsize=(10, 8))

    # Use the map projection natively stored in WRF (usually Lambert Conformal)
    # Using generic Lambert for cartopy here as an approximation for standard WRF grids
    proj = ccrs.LambertConformal(
        central_longitude=np.nanmean(lon_nr_trim),
        central_latitude=np.nanmean(lat_nr_trim)
    )

    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black',alpha=0.1,zorder=4)
    ax.add_feature(STATES,facecolor='none',edgecolor='black',linewidth=1,alpha=0.3,zorder=4)

    # Plotting NMEP (shaded)
    # Transform assumes data lat/lon are in PlateCarree
    levels = np.arange(0.1, 1.05, 0.05) # 10% to 100% probabilities
    nmep_fill = ax.contourf(
        lon_nr_trim, lat_nr_trim, resampled_nmep,
        levels=levels, cmap='afmhot_r', alpha=0.8,
        transform=ccrs.PlateCarree(), extend='max',zorder=3
    )

    # Outlining Nature Run truth exceedance (contour)
    ax.contour(
        lon_nr_trim, lat_nr_trim, uh_nr_trim,
        levels=[uh_threshold], colors='navy', linewidths=1,
        transform=ccrs.PlateCarree(), zorder=5
    )

    ax.set_extent(area)

    # Cosmetics
    cbar = plt.colorbar(nmep_fill, ax=ax, orientation='horizontal', pad=0.03, shrink=0.7)
    cbar.set_label(f"NMEP (UH > {uh_threshold:.1f})")

    # Added POD and FAR calculations into the title
    ax.set_title(
        f"UH Neighborhood Ensemble Probability\n"
        f"Truth ({PERCENTILE_THRESH}th percentile) | POD: {pod:.3f} | FAR: {far:.3f}",
        fontsize=12, fontweight='bold'
    )

    # Save and show
    output_filename = f"{rundir}/Figs/swaths/NMEP_vs_NR_{swt_var}_{exp}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_filename}")
    plt.show()

    return pod, far



def Performance_Diagram(experiments,pods,fars,PD_Fig_Title):
    pod_arr = np.linspace(0,1.1,1000)
    sr_arr = np.linspace(0,1.1,1000)

    x_sr,y_pod = np.meshgrid(sr_arr,pod_arr)
    csi_arr = (1/((1/x_sr) + (1/y_pod) -1))

    fig = plt.figure(figsize=(8,7))
    cols = ['k','r','g','b','brown','orchid','yellow']
    labs = experiments
    ax = plt.subplot(111)

    e = 0
    srs = []
    while e < len(experiments):
        sr = 1 - fars[e]
        ax.scatter(sr,pods[e],zorder=2,s=300,alpha=1,label=experiments[e])
        srs.append(sr)
        e+=1

    fill = ax.contourf(x_sr,y_pod,csi_arr,np.arange(0,0.85,0.05),cmap='Greys',alpha=0.8,zorder=1)

    ax.legend(loc='lower right',fontsize='large',framealpha=1,ncol=2)

    cb = fig.colorbar(fill,ax=ax,shrink=0.8,pad=0.02)
    cb.set_label(label='Critical Success Index (CSI)',size=20)
    cb.ax.tick_params(labelsize=16)

    ax.set_xlim(min(srs)-0.05,min(max(srs)+0.05,1))
    ax.set_ylim(min(pods)-0.1,min(max(pods)+0.05,1))

    ax.set_xlabel('Success Ratio (1-FAR)',fontsize=20)
    ax.set_ylabel('Probability of Detection (POD)',fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_title(f'{PD_Fig_Title[-4:]} Performance Diagram',fontsize=28)

    ax.grid(color='k',linestyle='--',alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{rundir}/Figs/perf_diagrams/{PD_Fig_Title}.png',facecolor='white')
    plt.show()
