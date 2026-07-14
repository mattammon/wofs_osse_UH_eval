import os
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, maximum_filter
import pyresample

from config import *

def subsection_idxs(area, lat, lon):
    """Find the array indices corresponding to a lat/lon bounding box."""
    lon_min, lon_max, lat_min, lat_max = area
    ilat = np.where((lat[:,0] >= lat_min) & (lat[:,0] <= lat_max))[0]
    ilon = np.where((lon[0,:] >= lon_min) & (lon[0,:] <= lon_max))[0]

    return ilat.min(), ilat.max(), ilon.min(), ilon.max()

def resample(data, old_lat, old_lon, new_lat, new_lon):
    """Resample data from the ensemble grid to the nature run grid."""
    orig_def = pyresample.geometry.SwathDefinition(lons=old_lon, lats=old_lat)
    targ_def = pyresample.geometry.SwathDefinition(lons=new_lon, lats=new_lat)

    resampled_variable = pyresample.kd_tree.resample_nearest(
        orig_def, data, targ_def, radius_of_influence=5000, fill_value=0
    )
    return resampled_variable

def create_circular_footprint(radius):
    """Generates a circular boolean footprint for the maximum_filter."""
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask

def process_member(member_num, threshold, ens_files_base, rad, swt_var):
    threshold = threshold/uh_factor[swt_var]
    """Accumulates UH and computes neighborhood exceedance for a single member."""
    mem_str = f"ENS_MEM_{member_num:02d}"

    # Generate this member's file list based on the base filenames
    files = [f.replace('ENS_MEM_01', mem_str) for f in ens_files_base]

    # 1. Accumulate UH for this member
    for i, file in enumerate(files):
        if not os.path.exists(file):
            print(f"Warning: Missing file {file}")
            return None

        with xr.open_dataset(file) as ds:
            if i == 0:
                uh = ds[ds_swt_var[swt_var]].data[0,:,:]
            else:
                uh = uh + ds[ds_swt_var[swt_var]].data[0,:,:]

    # 2. Binary thresholding using the nature run's percentile value
    binary_uh = (uh >= threshold).astype(float)

    # 3. Apply the neighborhood maximum filter
    footprint = create_circular_footprint(rad)
    neighborhood_uh = maximum_filter(binary_uh, footprint=footprint)

    return neighborhood_uh
