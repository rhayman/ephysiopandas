"""
You should make a copy of this file for each experiment analysis
and define global variables appropriately
"""

from enum import Enum, IntEnum
from pathlib import Path

# Define constants here
var_dict = {
    # ---- Paths to folders for loading, saving, plotting etc ----
    "data_dir": Path("data"),
    "git_dir": Path("gits"),
    "results_dir": Path("results"),
    "dataframes_dir": Path("dataframes"),
    "figures_dir": Path("figures"),
    # some output stuff
    "figure_format": "svg",
    # ---- general constants -----
    "map_binsize": 3.0,  # cm
    "map_smooth_size": 5,  # bins
    "pixels_per_metre": 300,
    "exclude_speeds": (0, 0.5),  # cm/s
    "high_speed_threshold": 20,  # cm/s
    "low_speed_threshold": 0.5,
    # -----linear track stuff-----
    "track_end_size": 14,  # cms - size of the track ends to be excluded
    "min_linear_track_spikes": 50,
    # final number of spikes for a cell to be considered for inclusion in
    # -----phase precession results-----
    "min_final_linear_track_spikes": 5,
    # minimum number of continuous bins for a place field on the linear track
    "min_linear_field_bins": 5,
    "min_runs": 3,  # for phase precession analysis
    # -----ratemap analysis-----
    # there are various arguments to ratemap creation not included here
    # see ephjysiopy.binning.ratemap for more details - keyword arguments
    # get passed down to the binning functions in there
    "min_firing_rate_threshold": 1,  # hz
    "max_firing_rate_threshold": 5,  # hz
    "min_spikes": 100,  # minimum number of spikes per cluster
    "min_theta_mod_spikes": 250,  # minimum num of spikes for theta modulation
    "field_threshold_percent": 150,
    "partition_method": "simple",
    # -----grid score method-----
    "expanding": True,  # whether to use the expanding circle sac method
    # -----LFP stuff-----
    "theta_band": (6, 12),  # hz
    "gamma_band": (20, 90),  # hz
    "slow_gamma": (20, 40),  # hz - same as jun et al., 2020
    "fast_gamma": (40, 90),  # hz - ditto
    "n_theta_bins": 36,
    # -----shuffle stuff-----
    "n_shuffles": 500,
    # -----threhsold stuff-----
    # percentile threhold for spatial information to be considered significant
    "spatial_info_threshold": 0.99,
    # hard-coded thresholds here should be adjusted based on generating
    # null distributions from shuffling data
    "threhsold_spatial_info": 0.5,
    # coherence threshold (from fenton et al., 2010)
    "threshold_coherence": 0.25,
}

figure_params = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "axes.grid": False,
    "font.size": 8,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "ytick.left": True,
    "ytick.direction": "in",
    "savefig.transparent": True,
    "savefig.bbox": "tight",
}


class RecordingType(Enum):
    Axona = 1
    OpenEphys = 2


class CellType(IntEnum):
    GRID = 1
    HD = 2
    BORDER = 3
    EGOCENTRIC_BOUNDARY = 4
    THETA = 5
    CENTRE_BEARING = 6
    CENTRE_DISTANCE = 7
    GRID_BY_HD = 8
    BORDER_BY_HD = 9
    HIGH_RATE_HD = 10
    HIGH_RATE = 11
    LOW_RATE = 12
    UNCLASSIFIED_HD = 13
    UNCLASSIFIED_SPATIAL = 14
    UNCLASSIFIED = 15
    SPEED_MODULATED = 16
