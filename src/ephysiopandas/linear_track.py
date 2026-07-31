"""
Analysis of a linear track data
"""

from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.utils import VariableToBin
from ephysiopy.phase_precession.linear_track import (
    run_phase_analysis,
    apply_linear_track_filter,
)
from ephysiopandas.config import var_dict

import ephysiopandas.data_io as data_io
import numpy as np
import pandas as pd
from pycircstat2.descriptive import circ_mean, circ_var
from pycircstat2.utils import rotate_data

EXCLUDE_SPEEDS = var_dict["exclude_speeds"]
TRACK_END_SIZE = var_dict["track_end_size"]
MAP_BINSIZE = var_dict["map_binsize"]
MIN_LINEAR_TRACK_SPIKES = var_dict["min_linear_track_spikes"]
MIN_LINEAR_FIELD_BINS = var_dict["min_linear_field_bins"]


def get_linear_bin_edges(
    trial: AxonaTrial, var2bin=VariableToBin.PHI, **kws
) -> list[np.ndarray, ...]:
    """
    Calculate equal bin edges for a single linear track trial

    Parameters
    ----------
    trial (AxonaTrial) - the trial

    **kws - keyword arguments
        contains things like binsize

    Returns
    -------
    list[np.ndarray] - the np.ndarray for the PHI bin edges

    Notes
    -----
    When creating the rate-maps we need to make sure
    the data is being binned into the same spatial bins
    so specify here the bin edges for the data to override
    the way this automatically gets calculated otherwise (see
    binning.py). Any filtering that has been applied to a trial
    is temporarily removed and the x-y positional extremes are
    used to calculate the bin edges. The filter(s) are then
    reapplied to the trial.
    """
    map_binsize = kws.get("map_binsize", 3)
    # get the filters for one of the trials...
    filters = trial.filter
    # ... and remove them
    trial.apply_filter()

    # get the bin edges for the unfiltered trial to apply
    # to both
    trial.RateMap.var2Bin = var2bin
    be = trial.RateMap._calc_bin_edges(map_binsize)

    # restore the filters
    trial.apply_filter(*filters)

    return [be]


def filter_trials_by_direction(
    T: AxonaTrial,
    var_type: VariableToBin = VariableToBin.X,
    **kws,
) -> tuple[AxonaTrial, AxonaTrial]:
    """
    Filters the trial into east (r or right) and
    west (l or left) bound runs

    Parameters
    ----------
    trial (AxonaTrial) - the trial to filter

    Returns
    -------
    tuple[AxonaTrial, AxonaTrial] - the trial filtered into
                                    right (Eastwards) and left (Westwards)
                                    bound runs
    """
    E = apply_linear_track_filter(T, "e", var_type, **kws)
    W = apply_linear_track_filter(T, "w", var_type, **kws)

    return E, W


def battaglia_overlap(
    left: AxonaTrial,
    right: AxonaTrial,
    bin_edges: list[np.ndarray],
    cluster: int,
    channel: int,
    **kws,
):
    """
    Replicates the overlap measure for place field similarity as per
    Battaglia et al., 2004.

    Notes
    -----
    Ruth used the unsmoothed ratemaps; it's not at all clear from the
    Battaglia paper whether they do or not but looking at the linear
    ratemaps in their Figure 2 it looks like they are smoothed

    I'm going to use smoothed ones...

    Parameters
    ----------
    left, right : AxonaTrial
        The two instances of AxonaTrial split into left/ right runs
    bin_edges : list[np.ndarray]
        The bin edges to use for the rate maps - must be equal
    cluster : int
        The cluster number
    channel : int
        The channel number
    kws : dict
        Additional keyword arguments to pass to get_rate_map()

    Returns
    -------
    a dict with keys/ values:
        shift_value (float) - the maximum overlap value
        shift_amount (int) - the amount of shift (in bins) to get the maximum
                            overlap value
    """

    left_map = left.get_rate_map(
        cluster,
        channel,
        var_type=VariableToBin.PHI,
        binsize=MAP_BINSIZE,
        bin_edges=bin_edges,
        smoothing=True,
        **kws,
    )
    right_map = right.get_rate_map(
        cluster,
        channel,
        var_type=VariableToBin.PHI,
        binsize=MAP_BINSIZE,
        bin_edges=bin_edges,
        smoothing=True,
        **kws,
    )

    def get_normed_map(A: np.ndarray) -> np.ndarray:
        A[np.isnan(A)] = 0
        return (len(A) * A) / np.nansum(A)

    # calculate the firing profile of the left and right maps as per
    # Battaglia et al.,
    P_left = get_normed_map(left_map.binned_data[0])
    P_right = get_normed_map(right_map.binned_data[0])
    P_left_right_sum = np.sum(P_left + P_right)
    # shift the right-hand ratemap to the left and right by 15 bins
    # to give a total of 31 shifts (inc. central, no shift one)
    shifts = np.arange(-15, 16, 1)
    overlap_scores = []
    for shift in shifts:
        # cyclically permute the right ratemap
        shifted_map = np.roll(right_map.binned_data[0], shift)
        shifted_map = get_normed_map(shifted_map)
        # calculate the overlap score
        score = 2 * (np.sum(np.fmin(shifted_map, P_left))) / P_left_right_sum

        overlap_scores.append(score)

    idx = np.argmax(overlap_scores)
    shift_amount = shifts[idx]
    shift_value = np.max(overlap_scores)

    return {"shift_value": shift_value, "shift_amount": shift_amount}


# --------------- PHASE PRECESSION ANALYSIS -------------------
def run_phase_precession_analysis(df: pd.DataFrame, **kws) -> pd.DataFrame:
    """
    Run the phase precession analysis on the linear track data
    and save the results to a dataframe

    There are many arguments that can be supplied to ephysiopy.phase_precession.run_phase_analysis

    Some of the important ones are:

        field_threshold=MIN_FIRING_RATE_THRESHOLD,
        field_threshold_percent=FIELD_THRESHOLD_PERCENT,
        minimum_allowed_run_speed=EXCLUDE_SPEEDS[1],
        track_end_size=TRACK_END_SIZE,
        partition_method="simple",
        return_pp=True,

    These should be supplied as a dict of kwargs loaded from ephysiopandas.config.var_dict
    i.e. as **kws to this function

    Returns
    -------
    pd.DataFrame - the dataframe containing the results

    Notes
    -----
    The dataframe will have the following columns:
    "filename" (str) - the name of the trial
    "chanel" (int) - tetrode number
    "cluster" (int) - cluster number
    "cell_type" (str) - cell type
    "field_id" (int) - field id (see ephysiopy.common.fieldcalcs.fieldprops)
    "slope" (float) - slope of the regression line
    "nspikes" (int) - number of spikes used in the analysis
    "intercept" (float) - y-intercept of the regression line
    "rho" (float) - correlation coefficient
    "pval" (float) - p-value from the shuffle test
    "phase_mean" (float) - circular mean of the spike phases (radians)
    "phase_variance" (float) - circular variance of the spike phases
    "field_size" (float) - size of the place field (cm)

    It's entirely possible that a cell has multiple place fields on the
    linear track so there may be multiple entries for a given cell

    """
    # Some of the merges create duplicate columns with _x and _y suffixes
    # so rename these back to the original name and then drop duplicates
    if "trial_type_x" in df.columns:
        df = df.rename(columns={"trial_type_x": "trial_type"})
    if "trial_type_y" in df.columns:
        df = df.rename(columns={"trial_type_y": "trial_type"})
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # assert "linear" in df.trial_type.unique()[0], (
    #     "Dataframe does not contain linear track data"
    # )

    pp_dict = dict.fromkeys(df.filename)

    run_dirs = ["w", "e"]

    output_df = pd.DataFrame(
        columns=[
            "filename",
            "channel",
            "cluster",
            "cell_type",
            "Genotype",
            "mouse_name",
            "run_direction",
            "n_spikes",
            "slope",
            "intercept",
            "rho",
            "pval",
            "phase_mean",
            "phase_variance",
            "pp_field_size",
        ]
    )

    for group in df.groupby("filename"):
        pname = group[0]
        if "linear" in str(pname):
            T = data_io.load_trial(pname)

            for row in group[1].iterrows():
                cluster = row[1]["cluster"]
                channel = row[1]["channel"]
                cell_type = row[1]["cell_type"]
                mouse_name = row[1]["mouse_name"]
                phenotype = row[1]["Genotype"]
                n_spikes_west = row[1]["num_spikes_west"]
                n_spikes_east = row[1]["num_spikes_east"]
                # skip if there are insufficient spikes in either direction
                if np.logical_or(
                    n_spikes_west <= MIN_LINEAR_TRACK_SPIKES,
                    n_spikes_east <= MIN_LINEAR_TRACK_SPIKES,
                ):
                    continue
                pp_dict[pname] = {}
                pp_dict[pname][(cluster, channel)] = {}
                for run_dir in run_dirs:
                    pp_dict[pname][(cluster, channel)][run_dir] = {}
                    try:
                        pp_results, PP = run_phase_analysis(
                            T,
                            int(cluster),
                            int(channel),
                            run_direction=run_dir,
                            return_pp=True,
                            **kws,
                        )
                    except Exception:
                        pp_results = None
                    if pp_results:
                        pp_dict[pname][(cluster, channel)
                                       ][run_dir] = pp_results
                        # insert the results into the dataframe
                        for i, result in enumerate(pp_results):
                            # ignore nans etc
                            if np.isfinite(result.stats.slope):
                                fp = PP.field_properties[i]
                                if fp.area >= MIN_LINEAR_FIELD_BINS:
                                    num_spikes = fp.n_spikes
                                    phase = fp.spike_phase.ravel()
                                    # rotate the phases so that the circular
                                    # mean is at pi
                                    phase = rotate_data(phase, np.pi)

                                    phase_mean = circ_mean(phase)
                                    phase_var = circ_var(phase)

                                    # get field size in cm
                                    field_size = np.abs(
                                        np.diff(fp.bin_coords[0][[0, -1]])
                                    )[0]

                                    output_df = pd.concat(
                                        [
                                            pd.DataFrame(
                                                [
                                                    [
                                                        pname,
                                                        channel,
                                                        cluster,
                                                        cell_type,
                                                        phenotype,
                                                        mouse_name,
                                                        run_dir,
                                                        num_spikes,
                                                        result.stats.slope,
                                                        result.stats.intercept,
                                                        result.stats.rho,
                                                        result.stats.p_shuffled,
                                                        phase_mean,
                                                        phase_var,
                                                        field_size,
                                                    ]
                                                ],
                                                columns=output_df.columns,
                                            ),
                                            output_df,
                                        ],
                                        ignore_index=True,
                                    )

    if kws.get("save_as", None):
        output_df.to_pickle(kws["save_as"])

    return output_df
