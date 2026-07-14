import os
from config import *
from main import main, Performance_Diagram

##################################
date = '20250314'

experiments = ['SfcRad', 'SfcRad_UAS', 'SfcRad_150UAS',
               'SfcRad_150UAS_XLV', 'SfcRad_75UAS', 'SfcRad_20UAS', 'SfcRad_20UAS_XLV']

# experiments = ['SfcRad', 'SfcRad_UAS', 'SfcRad_20UAS','SfcRad_20UAS_largeH', 'SfcRad_20UAS_largeV',
#                'SfcRad_20UAS_midH', 'SfcRad_20UAS_midHlargeV', 'SfcRad_20UAS_smHlgV',
#                'SfcRad_20UAS_XLV']

# aliases = ['SfcRad', 'SfcRad_UAS', '20UAS_70H-0.5V','20UAS_250H-0.5V', '20UAS_70H-1V',
#                '20UAS_140H-0.5V', '20UAS_140H-1V', '20UAS_35H-1V',
#                '20UAS_70H-2V']

# experiments = ['SfcRad', 'SfcRad_UAS', 'SfcRad_UAS_all', 'SfcRad_IRS', 'SfcRad_IRS_lidar',
#                'SfcRad_lidar', 'SfcRad_perfect']
variable = 'UH25'
params = {
    'NUM_MEMBERS':18,
    'PERCENTILE_THRESH':95.0,
    'NEIGHBORHOOD_RADIUS':5,
    'SMOOTHING_SIGMA':2
}
title = f'{date}_NUM-SITES_{variable}'
##################################

pods = []
fars = []

for e in experiments:
    # --- Configuration ---
    ENS_DIR = f'/scratch/wofs/matthew.ammon/nature_runs/fcst/{date}/{e}/{fcst_stm[date]}'
    if not os.path.isdir(ENS_DIR):
        ENS_DIR = f'/work/joshua.gebauer/Big_NatureRun/fcst/{date}/{e}/{fcst_stm[date]}'
    NR_DIR = f'/work2/wof/realtime/nature_run/{date}/nr_ens0'

    POD, FAR = main(e,variable,ENS_DIR,NR_DIR,**params)
    pods.append(POD)
    fars.append(FAR)

Performance_Diagram(experiments,pods,fars,title)
