import numpy as np
import pandas as pd
from functools import lru_cache, cache
from pycircstat2.hypothesis import rayleigh_test
from pycircstat2.utils import rotate_data
from pycircstat2.descriptive import circ_mean, circ_var, circ_r
from ephysiopy.io.bases import TrialInterface as Trial
from ephysiopy.common.utils import MapType, corr_maps, filter_trial_by_time
import ephysiopy.common.fieldcalcs as fc
from ephysiopy.common.waveformcalcs import (
    peak_local_max_time,
)
from ephysiopandas.data_io import (
    load_trial,
    get_SpikeCalcsGeneric,
    get_WaveformCalcs,
    get_LFPOscillation,
)
from ephysiopandas.helpers import (
    get_environment_shape,
    get_field_props,
    get_environment_type,
    print_measure_details,
    single_or_list,
    assert_columns_exist,
)
from ephysiopandas.cell_type_classifier import get_labels


def is_cluster_good(trial: Trial, cluster: int, channel: int) -> bool:
    """
    Returns True if the given cluster and channel in the given trial
    is marked as 'good', False otherwise.
    """
    if trial.clusterData or trial.load_cluster_data():
        return cluster in trial.clusterData.good_clusters
    else:
        return True  # Axona data without cluster quality info


def classify_cell_type(df: pd.DataFrame, **kws) -> pd.DataFrame:
    """
    Classify cells as either Interneuron or Pyramidal based solely
    on their peak to trough time
    """
    assert_columns_exist(df, ["spike_width"])
    df["cell_type"] = get_labels(np.array(df.spike_width))
    return df


@print_measure_details
def num_spikes(trial: Trial, cluster: int, channel: int, **kws) -> int:
    """
    Returns the number of spikes for the given cluster and channel
    in the given trial.
    """
    spike_times = trial.get_spike_times(cluster, channel)
    return len(spike_times)


@print_measure_details
def overdispersion(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculates an overdispersion measure for the provided cluster
    following the method of Fenton et al., 2010.

    Notes
    -----
    Periods where the expected number of spikes is less than 5 are
    removed from the calculation.
    """
    if not trial.PosCalcs:
        trial = load_trial(trial.pname)

    binned_data = trial.get_rate_map(
        cluster,
        channel,
        **kws,
    )

    xy = trial.PosCalcs.xy

    binned_spikes = trial.get_binned_spike_times(cluster, channel)

    od, _ = fc.fast_overdispersion(binned_data, xy, binned_spikes)

    return od


@single_or_list
@print_measure_details
def peak_rate(trial: Trial, cluster: int, channel: int, **kws) -> list:
    """
    Get the peak firing rate for a given cluster and channel.
    This is the maximum firing rate calculated from the spike times.
    """

    ratemap = trial.get_rate_map(
        cluster,
        channel,
        **kws,
    )

    peak_rates = []

    for rm in ratemap:
        peak_rates.append(np.nanmax(rm.binned_data[0]))

    return peak_rates


@print_measure_details
def mean_rate(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Get the mean firing rate for a given cluster and channel.
    This is the total number of spikes divided by the duration of the trial.
    """

    spike_times = trial.get_spike_times(cluster, channel)
    duration = trial.PosCalcs.duration

    mean_rate = len(spike_times) / duration

    return mean_rate


@single_or_list
@print_measure_details
def coherence(trial: Trial, cluster: int, channel: int, **kws) -> list:
    """
    Calculate the coherence score for a given cluster in the trial.
    """

    ratemap = trial.get_rate_map(
        cluster,
        channel,
        **kws,
    )
    ratemap_unsmoothed = trial.get_rate_map(
        cluster,
        channel,
        smoothing=False,
        **kws,
    )

    coherence_scores = []

    for rm, rm_unsm in zip(ratemap, ratemap_unsmoothed):
        coherence_scores.append(fc.coherence(rm.binned_data[0], rm_unsm.binned_data[0]))

    return coherence_scores


@single_or_list
@print_measure_details
def spatial_sparsity(trial: Trial, cluster: int, channel: int, **kws) -> list:
    """
    Calculate the spatial sparsity score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    ratemap = trial.get_rate_map(
        cluster,
        channel,
        **kws,
    )

    pos_map = trial.get_rate_map(
        cluster,
        channel,
        map_type=MapType.POS,
        smoothing=False,
        **kws,
    )

    sparsity_scores = []

    for rm, pm in zip(ratemap, pos_map):
        sparsity_scores.append(
            fc.spatial_sparsity(rm.binned_data[0], pm.binned_data[0])
        )

    return sparsity_scores


@single_or_list
@print_measure_details
def kl_spatial_sparsity(trial: Trial, cluster: int, channel: int, **kws) -> list:
    """
    Calculate the KL spatial sparsity score for a given trial.
    Note this is a measure of how well sampled the environment is compared
    to a uniform distribution.
    """

    pos_map = trial.get_rate_map(
        cluster,
        channel,
        map_type=MapType.POS,
        smoothing=False,
        **kws,
    )

    sparsity_scores = []

    for pm in pos_map:
        sparsity_scores.append(fc.kl_spatial_sparsity(pm))

    return sparsity_scores


@print_measure_details
def theta_mod_v1(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the theta modulation score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    S = get_SpikeCalcsGeneric(trial, cluster, channel)
    return S.theta_mod_idx()


@print_measure_details
def theta_mod_v2(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the theta modulation score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    S = get_SpikeCalcsGeneric(trial, cluster, channel)
    return S.theta_mod_idxV2()


@print_measure_details
def theta_mod_v3(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the theta modulation score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    S = get_SpikeCalcsGeneric(trial, cluster, channel)
    return S.theta_mod_idxV3(**kws)


@print_measure_details
def theta_band_max_freq(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the maximum theta band frequency for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    S = get_SpikeCalcsGeneric(trial, cluster, channel)
    return S.theta_band_max_freq(**kws)


@single_or_list
@print_measure_details
def grid_score(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the grid score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    ratemap = trial.get_grid_map(
        cluster,
        channel,
        map_type=MapType.RATE,
        smoothing=True,
        **kws,
    )

    if "expanding" in kws.keys():
        if kws["expanding"] is True:
            return fc.expanding_circle_gridscore(ratemap.binned_data[0])
        else:
            return fc.basic_gridscore(ratemap.binned_data[0])
    else:
        return fc.basic_gridscore(ratemap.binned_data[0])


@single_or_list
@print_measure_details
def head_directionality(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the head directionality score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    ratemap = trial.hd_map(
        cluster,
        channel,
        **kws,
    )

    head_scores = []

    for rm in ratemap:
        head_scores.append(fc.kldiv_dir(ratemap.binned_data[0]))

    return head_scores


@print_measure_details
def speed_rate_correlation(trial: Trial, cluster: int, channel: int, **kws) -> dict:
    """
    Calculate the speed-rate correlation for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    ts = trial.get_spike_times(cluster, channel)
    speed = trial.PosCalcs.speed

    S = get_SpikeCalcsGeneric(trial, cluster, channel)

    pearson_result = S.ifr_sp_corr(ts, speed)

    return {
        "pearson_statistic": pearson_result.statistic,
        "pearson_pvalue": pearson_result.pvalue,
    }


@single_or_list
@print_measure_details
def spatial_info(trial: Trial, cluster: int, channel: int, **kws) -> list:
    """
    Calculate the spatial information score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    ratemap = trial.get_rate_map(
        cluster,
        channel,
        map_type=MapType.ADAPTIVE,
        smoothing=False,
        **kws,
    )
    pos_map = trial.get_rate_map(
        cluster,
        channel,
        map_type=MapType.POS,
        smoothing=False,
        **kws,
    )

    spatial_information_scores = []
    for rm in ratemap:
        spatial_information_scores.append(
            fc.skaggs_info(rm.binned_data[0], pos_map.binned_data[0])
        )

    return spatial_information_scores


@single_or_list
@print_measure_details
def border_score(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the border score for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """
    ratemap = trial.get_rate_map(
        cluster,
        channel,
        **kws,
    )

    shape = get_environment_shape(trial.PosCalcs.xy.T)

    return fc.border_score(ratemap.binned_data[0], shape=shape)


@single_or_list
@print_measure_details
def head_dir_kl_div(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Calculate the Kullback-Leibler divergence for head directionality
    for a given cluster in the trial.
    """

    ratemap = trial.get_hd_map(
        cluster,
        channel,
        **kws,
    )

    scores = []

    for rm in ratemap:
        scores.append(fc.kldiv_dir(ratemap.binned_data[0]))

    return scores


@print_measure_details
def head_dir_mean_resultant_length(
    trial: Trial, cluster: int, channel: int, **kws
) -> float:
    """
    Calculate the mean resultant length for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """
    # need to make sure the hd map is calculated and cached before
    # calculating the mean resultant length
    _ = trial.get_hd_map(
        cluster,
        channel,
        **kws,
    )
    spiked_at = trial.RateMap.get_samples_when_spiking()

    return circ_r(np.deg2rad(spiked_at))


@print_measure_details
def head_dir_mean_resultant_angle(
    trial: Trial, cluster: int, channel: int, **kws
) -> float:
    """
    Calculate the mean resultant angle for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """
    # need to make sure the hd map is calculated and cached before
    # calculating the mean resultant length
    _ = trial.get_hd_map(
        cluster,
        channel,
        **kws,
    )
    spiked_at = trial.RateMap.get_samples_when_spiking()

    return circ_mean(np.deg2rad(spiked_at))


@print_measure_details
def spike_width(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Get the time between the peak and trough of the average waveform
    """
    S = get_WaveformCalcs(trial, cluster, channel)
    best_channel = S.get_best_channel()
    mean_waveform, _ = S.mean_waveform(best_channel)

    if mean_waveform.shape[0] == 50:
        fs = 5e4  # 50 samples at 50 kHz
    else:
        fs = 3e4  # 80 samples at 30 kHz

    p2t = peak_local_max_time(mean_waveform, fs=int(fs))
    # p2t = peak_to_trough_time(mean_waveform, fs=fs)

    return p2t["peak_to_trough_ms"] * 1000  # convert to microseconds


@print_measure_details
def waveform_inverted(trial: Trial, cluster: int, channel: int, **kws) -> bool:
    """
    Get whether the waveform is inverted for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    S = get_WaveformCalcs(trial, cluster, channel)
    best_channel = S.get_best_channel()
    mean_waveform, _ = S.mean_waveform(best_channel)

    if mean_waveform.shape[0] == 50:
        fs = 5e4  # 50 samples at 50 kHz
    else:
        fs = 3e4  # 80 samples at 30 kHz

    p2t = peak_local_max_time(mean_waveform, fs=int(fs))

    return p2t["inverted"]


@print_measure_details
def ahp_decay(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Get the afterhyperpolarization (AHP) decay time for a given cluster in the
    trial.
    """

    W = get_WaveformCalcs(trial, cluster, channel)

    ahp = W.estimate_AHP()

    if ahp == 0:  # waveforms might be inverted due to reference electrode
        W.invert_waveforms = True
        ahp = W.estimate_AHP()

    return ahp


@print_measure_details
def contamination_Q(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Get the contamination Q for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    return contamination_percent(trial, cluster, channel)[0]


@print_measure_details
def contamination_R(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Get the contamination Q for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    return contamination_percent(trial, cluster, channel)[1]


@cache
def contamination_percent(
    trial: Trial, cluster: int, channel: int, **kws
) -> tuple[float, float]:
    """
    Get the contamination percentage for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    S = get_SpikeCalcsGeneric(trial, cluster, channel)

    return S.contamination_percent()


@print_measure_details
def waveform_property(
    trial: Trial, cluster: int, channel: int, prop: str, **kws
) -> float:
    """
    Get a waveform property for a given cluster in the trial.
    This is a placeholder function, as the actual calculation
    would depend on the specific implementation details.
    """

    S = get_SpikeCalcsGeneric(trial, cluster, channel)

    return S.get_waveform_property(prop)


@print_measure_details
def phase_locking_pval(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Returns the p-value for the rayleigh test for the phase
    at which the spikes occur in the theta cycle.
    """
    _, _, pval = phase_locking_values(trial, cluster, channel)
    return pval


@print_measure_details
def phase_locking_z_stat(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Returns the rayleigh Z statistic for the phase
    at which the spikes occur in the theta cycle.
    """
    _, z, _ = phase_locking_values(trial, cluster, channel)
    return z


@print_measure_details
def phase_locking_vector_length(
    trial: Trial, cluster: int, channel: int, **kws
) -> float:
    """
    Returns the mean resultant vector length for the phase
    at which the spikes occur in the theta cycle.
    """
    r, _, _ = phase_locking_values(trial, cluster, channel)
    return r


@lru_cache(maxsize=100, typed=False)
def phase_locking_values(
    trial: Trial, cluster: int, channel: int, **kws
) -> tuple[float, ...]:
    """
    Returns the mean resultant vector length, the rayleigh Z statistic and the
    p-value for the rayleigh test for the phase
    at which the spikes occur in the theta cycle.
    """
    L = get_LFPOscillation(trial)
    phase, _, _ = L.get_theta_phase(trial.get_spike_times(cluster, channel))
    # shift phase to be in [0, 2pi] range
    phase = rotate_data(phase, np.pi)
    result = rayleigh_test(phase)
    return result.r, result.z, result.pval


@print_measure_details
def phase_mean(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Returns the mean and variance of the phase of firing
    at which the spikes occur in the theta cycle.
    """
    mean, _ = phase_values(trial, cluster, channel)
    return mean


@print_measure_details
def phase_variance(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Returns the mean and variance of the phase of firing
    at which the spikes occur in the theta cycle.
    """
    _, var = phase_values(trial, cluster, channel)
    return var


@lru_cache(maxsize=100, typed=False)
def phase_values(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Returns the mean and variance of the phase of firing
    at which the spikes occur in the theta cycle.
    """
    L = get_LFPOscillation(trial)
    phase, _, _ = L.get_theta_phase(trial.get_spike_times(cluster, channel))
    # shift phase to be in [0, 2pi] range
    phase = rotate_data(phase, np.pi)
    return circ_mean(phase), circ_var(phase)


@print_measure_details
def phase_distribution(trial: Trial, cluster: int, channel: int, **kws) -> np.ndarray:
    """
    Returns the phase distribution of the spikes in the theta cycle.
    """
    nbins = kws.get("nbins", 36)

    L = get_LFPOscillation(trial)
    phase, _, _ = L.get_theta_phase(trial.get_spike_times(cluster, channel))
    # shift phase to be in [0, 2pi] range
    phase = rotate_data(phase, np.pi)
    hist, _ = np.histogram(phase, bins=nbins, range=(0, 2 * np.pi), density=True)
    # ensure that we return a probability mass function
    hist = hist / np.sum(hist)
    return hist


@print_measure_details
def field_size(trial: Trial, cluster: int, channel: int, **kws) -> float:
    """
    Get the field size in cms for the cluster
    """

    # get the ennvironment type (linear or 2D) and get
    # the appropriate rate map
    env = get_environment_type(trial.PosCalcs.xy)

    kws["partition_method"] = "simple" if env == "linear" else "fancy"

    # Get the field properties
    # it's possible that there are no fields, so handle that
    try:
        fp = get_field_props(
            trial,
            cluster,
            channel,
            env=env,
            **kws,
        )
    except Exception:
        return np.nan

    if len(fp) == 0:
        return np.nan
    # sort the fields by size
    fp = fc.sort_fields_by_attr(fp, attr="area")

    # Get the mean bin size averaging across bin sizes
    bd = fp[0].binned_data

    if "linear" in env:
        xm = np.mean(np.diff(bd.bin_edges[0]))
    else:
        xm = np.mean(np.diff(bd.bin_edges[1]))

    ym = np.mean(np.diff(bd.bin_edges[0]))

    binsize = np.mean([xm, ym])

    return fp[0].area * binsize


# @print_measure_details
def within_trial_cluster_correlations(
    trial: Trial, cluster: int, channel: int, how: str = "in_half", **kws
) -> float:
    """
    Performs within trial correlations of the ratemaps for each recorded
    cluster within a trial.

    Trials are split by "how" and correlations are performed on
    the rate maps generated from each split

    Parameters
    ----------
    trial : str
        'open' or 'linear'
    cell_type : str
        'Pyramidal' or 'Interneuron'
    how : str
        'in_half' or 'odd_even'
    """
    if not trial.PosCalcs:
        trial = load_trial(trial.pname)

    xlims = trial.RateMap._x_lims
    ylims = trial.RateMap._y_lims

    filt1, filt2 = filter_trial_by_time(trial.PosCalcs.duration, how)

    trial.apply_filter(*filt1)
    trial.RateMap._x_lims = xlims
    trial.RateMap._y_lims = ylims

    m1 = trial.get_rate_map(cluster, channel)

    trial.apply_filter(*filt2)
    trial.RateMap._x_lims = xlims
    trial.RateMap._y_lims = ylims

    m2 = trial.get_rate_map(cluster, channel)

    return corr_maps(m1.binned_data[0], m2.binned_data[0])
