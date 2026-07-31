from pathlib import Path
import warnings
from functools import lru_cache
import copy
import numpy as np
from scipy import signal
import pandas as pd
from ephysiopy.io.recording import (
    AxonaTrial,
    OpenEphysAcqBoard,
    OpenEphysFPGA,
)
from ephysiopy.openephys2py.OESettings import Settings
from ephysiopy.io.bases import TrialInterface
from ephysiopy.common.utils import TrialFilter, filter_trial_by_time
from ephysiopy.common.spikingcalcs import SpikeCalcsGeneric
from ephysiopy.common.waveformcalcs import WaveformCalcsGeneric
from ephysiopy.common.phasecoding import LFPOscillations
from ephysiopandas.config import RecordingType


def save_dataframe(df: pd.DataFrame, fname: str, format="pkl") -> None:
    """
    Saves a DataFrame to disk as a pickle file.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    fname : str
        The filename (including path) to save the DataFrame to.
    """
    save_path = Path(fname)
    if format == "pkl":
        df.to_pickle(save_path)
    elif format == "csv":
        df.to_csv(save_path, index=False)
    elif format == "hdf":
        df.to_hdf(save_path, key="df", mode="w")
    else:
        raise ValueError(f"Unsupported format: {format}")
    print(f"Saved DataFrame to {save_path}")
    return


def load_dataframe(fname: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a pickle file on disk.
    Parameters
    ----------
    fname : str
        The filename (including path) to load the DataFrame from.
    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    load_path = Path(fname)
    assert load_path.exists()
    suffix = load_path.suffix
    if suffix == ".pkl":
        df = pd.read_pickle(load_path)
    elif suffix == ".csv":
        df = pd.read_csv(load_path)
    elif suffix == ".hdf":
        df = pd.read_hdf(load_path, key="df")
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    print(f"Loaded DataFrame from {load_path}")
    return df


# Scan for files and populate a DataFrame
def scan_drive(
    src_dir: Path, out_dir: Path, update=False, concatenated=False
) -> pd.DataFrame:
    """
    Scans the given source directory for Axona .set files
    and OpenEphys recording directories, creating a DataFrame
    that maps each trial to its recording type.

    Parameters
    ----------
    src_dir : PosixPath
        The source directory to scan for recording files.
    output_dir : PosixPath
        The directory where the resulting DataFrame will be saved.
    update : bool
        If True, forces a rescan of the source directory
        and updates the DataFrame. If False, loads the existing
        DataFrame from disk if it exists.
    concat : bool
        If True, the data to be loaded is assumed to be a pickle
        file which is the result of loading two or more files and
        combining them into a single trial.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: 'Trial' and 'Recording',
        where 'Trial' is the path to the recording file or directory,
        and 'Recording' indicates the type of recording
        (Axona or OpenEphys).
    """
    assert src_dir.exists()

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    if update:
        if not concatenated:
            # Axona is easiest to find due to .stm file requirement
            stm_files = src_dir.rglob("*.stm")
            set_files = [str(file.with_suffix(".set")) for file in stm_files]
            # OE files/ directories
            ttl_dirs = src_dir.rglob("TTL")
            oe_dirs = [
                str(d.parents[5]) for d in ttl_dirs if Path(d.parents[5]).exists()
            ]
            # create a dict ready to make the DataFrame
            d1 = dict.fromkeys(set_files, RecordingType.Axona)
            d2 = dict.fromkeys(oe_dirs, RecordingType.OpenEphys)
            d3 = dict(d1, **d2)
            # convert the filenames from strings to Paths
            d4 = {Path(k[0]): k[1] for k in d3.items()}
            # save and return the DataFrame
            df = pd.DataFrame(
                data=d4.items(), index=range(len(d4)), columns=["filename", "Recording"]
            )
            save_dataframe(df, out_dir / Path("file_location.pkl"))
            return df
        else:
            pkls = src_dir.rglob("*.pkl")
            rec_type = []
            fnames = []
            for pkl in pkls:
                fnames.append(pkl)  # make sure same order as rec_type
                trial = load_trial(pkl)
                # attempt to access TETRODE - if present then Axona if not OE
                try:
                    trial.TETRODE
                    rec_type.append(RecordingType.Axona)
                except Exception:
                    rec_type.append(RecordingType.OpenEphys)
            d = {f: r for f, r in zip(fnames, rec_type)}
            df = pd.DataFrame(
                data=d.items(),
                index=range(len(fnames)),
                columns=["filename", "Recording"],
            )
            save_dataframe(df, out_dir / Path("file_location.pkl"))
            return df

    else:
        assert (out_dir / Path("file_location.pkl")).exists()
        return load_dataframe(out_dir / Path("file_location.pkl"))


def create_cluster_df(
    df: pd.DataFrame, out_name: Path, update: bool = False
) -> pd.DataFrame:
    """
    Creates a DataFrame listing all clusters and tetrodes
    for each trial in the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with columns 'Trial' and 'Recording'.
    out_name : PosixPath
        The name the resulting cluster DataFrame will be saved/ loaded as.
    update : bool
        If True, forces a rescan of the trials to populate
        the cluster DataFrame. If False, loads the existing
        cluster DataFrame from disk if it exists.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'filename', 'tetrode', and 'cluster',
        listing all clusters and tetrodes for each trial.
    """
    if update:
        trials = df.filename.tolist()
        channels = []
        clusters = []
        filenames = []
        for trial_name in trials:
            trial = load_trial(Path(trial_name))
            channels_clusters = trial.get_available_clusters_channels()
            for channel in channels_clusters.keys():
                for cluster in channels_clusters[channel]:
                    if filter_for_good(trial, cluster):
                        channels.append(channel)
                        clusters.append(cluster)
                        filenames.append(trial_name)
        cluster_df = pd.DataFrame(
            {
                "filename": filenames,
                "channel": channels,
                "cluster": clusters,
            }
        )
        # get the format from the suffix, e.g. 'pkl'
        format = out_name.suffix[1:]  # remove the dot
        save_dataframe(cluster_df, out_name, format=format)
    else:
        cluster_df = load_dataframe(out_name)

    return cluster_df


@lru_cache(maxsize=8)
def load_trial(pname: Path, **kws) -> TrialInterface:
    """
    Loads an TrialInterface (AxonaTrial or OpenEphysBase) object
    from its' pathname, taking
    account of the correct pixels_per_metre value and applying
    all the necessary initialisations and settings.

    Parameters
    ----------
    pname : PosixPath
        The absolute path to the .set file
    **kws : dict
        keyword arguments:

            pixels_per_metre, default = 665
            map_binsize, default = 3
            map_smooth_size, default = 5
            exclude_speeds, default = (0, 0.5)

    Returns
    -------
    AxonaTrial | OpenEphysBase
        The loaded and initialised AxonaTrial or OpenEphysBase object
    """
    # parse kws
    PPM = kws.get("pixels_per_metre", 665)
    MAP_BINSIZE = kws.get("map_binsize", 3)
    SMOOTH_SIZE = kws.get("map_smooth_size", 5)
    EXCLUDE_SPEEDS = kws.get("exclude_speeds", (0, 0.5))

    pname = Path(pname)

    # if it's a pickle, load the trial from the pickle and return it
    # i.e. assume the pre-processing has happened to the concatenated trials
    if pname.suffix == ".pkl":
        with open(pname, "rb") as f:
            T = pd.read_pickle(f)
            T.initialise()
    elif pname.suffix == ".set":
        T = AxonaTrial(pname)
        PPM = int(T.settings["tracker_pixels_per_metre"])
    elif pname.is_dir():
        # scan the directory for the relevant recording type
        # based on the plugins described in the settings.xml file
        settings = Settings(pname)
        processors = list(settings.processors.keys())
        processors = [
            p.split(" ")[0] for p in processors
        ]  # get the first word of each processor name

        if "Acquisition" in processors:
            T = OpenEphysAcqBoard(pname, **kws)
        elif "FPGA" in processors:
            T = OpenEphysFPGA(pname, **kws)
        else:
            raise ValueError(
                f"Unknown OpenEphys recording type in {pname}. "
                f"Processors found: {processors}"
            )
        T.load_cluster_data(**kws)

    T.load_pos_data(ppm=PPM)

    try:
        # provide parameters for OE TTL loading, ignored for Axona
        T.load_ttl(StimControl_id="StimControl 110", TTL_channel_number=1)
    except Exception:
        pass

    # filter out low run speeds
    filter = TrialFilter("speed", EXCLUDE_SPEEDS[0], EXCLUDE_SPEEDS[1])
    T.apply_filter(filter)

    T.initialise()
    T.RateMap.smooth_sz = SMOOTH_SIZE
    T.RateMap.binsize = MAP_BINSIZE
    T.RateMap.pos_weights = np.ones_like(T.RateMap.dir)

    return T


def filter_for_data(trialname: Path) -> bool:
    """
    Checks if the trial at the given pathname has data.

    Parameters
    ----------
    trialname : PosixPath
        The path to the trial file or directory.

    Returns
    -------
    bool
        True if the trial has spike data, False otherwise.
    """
    trial = load_trial(trialname)
    return [True if trial.ttl_data else False][0]


def filter_for_good(trial: TrialInterface, cluster: int) -> bool:
    """
    Filters for 'good' labels in KiloSort/ phy metadata

    If Axona returns True

    Parameters
    ----------
    trial : TrialInterface
        The trial to examine

    cluster : int
        The cluster to check
    """
    if ".set" == trial.pname.suffix:
        return True

    if not trial.template_model:
        trial.load_neural_data()

    assert cluster in trial.template_model.cluster_ids

    user_labels = trial.template_model.metadata["KSLabel"]

    try:
        user_labels = trial.template_model.metadata["group"]
    except KeyError:
        warnings.warn("No user defined groups. Falling back on KSLabels")

    return "good" in user_labels[cluster]


def filter_trials_by_time(
    trial: TrialInterface, how="in_half"
) -> tuple[TrialInterface, ...]:
    """
    Filters the data in trial by time

    Parameters
    ----------
    trial (TrialInterface) - the trial object

    how (str) - how to split the trial.
                Legal values: "in_half" or "odd_even"
                "in_half" filters for first n seconds and last n second
                "odd_even" filters for odd vs even minutes

    Returns
    -------
    A pair of TrialInterface instances filtered for time
    """
    assert how in ["in_half", "odd_even"]

    duration = trial.PosCalcs.duration

    f1, f2 = filter_trial_by_time(duration, how)

    t1 = copy.deepcopy(trial)
    t2 = copy.deepcopy(trial)

    t1.apply_filter(*f1)
    t2.apply_filter(*f2)

    return t1, t2


def extract_mean_waveform(
    trial: TrialInterface, cluster: int, channel: int, **kws
) -> np.ndarray:
    """
    Extracts the mean waveform for the given trial, cluster, and channel.

    Parameters
    ----------
    trial : TrialInterface
        The trial to extract the waveform from.
    cluster : int
        The cluster to extract the waveform for.
    channel : int
        The channel to extract the waveform from.

    Returns
    -------
    np.ndarray
        The mean waveform for the given trial, cluster, and channel.

    Notes
    -----
    The OpenEphys template_gui package pulls out a longer sample than Axona so that
    is truncated here to match the length of the Axona waveforms (50 samples),
    so that PCA can be performed on the result(s)
    """
    W = get_WaveformCalcs(trial, cluster, channel, **kws)
    best_channel = W.get_best_channel()
    mean_waveform, _ = W.mean_waveform(best_channel)

    if np.shape(mean_waveform)[-1] > 50:
        # get the 1ms more or less corresponding to Axona
        # i.e. 200 microsecond pre spike, 800 post
        mean_waveform = mean_waveform[30:63]
        # resample to match 50 samples
        mean_waveform = signal.resample_poly(mean_waveform, 50, 33, axis=-1)
        # multiply by 0.159 to match the microvolt scaling of Axona waveforms
        mean_waveform = mean_waveform * 0.159
        # get into microvolts to match Axona waveforms
        mean_waveform = mean_waveform / 1e6

    return mean_waveform


@lru_cache(maxsize=8)
def get_WaveformCalcs(
    trial: TrialInterface, cluster: int, channel: int, **kws
) -> WaveformCalcsGeneric:
    """
    Returns a SpikeCalcsGeneric object containing the waveform data for the given trial, cluster, and channel.
    """
    W = WaveformCalcsGeneric(
        trial.get_waveforms(cluster, channel, from_raw=True),
        trial.get_spike_times(cluster, channel),
        cluster,
    )

    # check if Axona, if not change sample rate etc
    try:
        trial.TETRODE
    except AttributeError:
        W.sample_rate = trial.template_model.sample_rate
        W.pre_spike_samples = 41
        W.post_spike_samples = 41

    return W


@lru_cache(maxsize=8)
def get_SpikeCalcsGeneric(
    trial: TrialInterface, cluster: int, channel: int
) -> SpikeCalcsGeneric:
    """
    Returns a SpikeCalcsGeneric object
    """
    S = SpikeCalcsGeneric(
        trial.get_spike_times(cluster, channel),
        cluster,
        trial.get_waveforms(cluster, channel),
    )

    # check if Axona, if not change sample rate etc
    try:
        trial.TETRODE
    except AttributeError:
        S.sample_rate = trial.template_model.sample_rate
        S.pre_spike_samples = 41
        S.post_spike_samples = 41

    try:
        S.event_ts = trial.ttl_data["ttl_timestamps"]
    except TypeError:
        warnings.warn("No TTL data present")

    S.duration = trial.PosCalcs.duration
    return S


@lru_cache(maxsize=8)
def get_LFPOscillation(trial: TrialInterface, *args) -> LFPOscillations:
    """
    Returns a LFPOscillations object for the given trial, cluster, and channel.
    """

    if not trial.EEGCalcs:
        trial.load_lfp(*args)

    return LFPOscillations(trial.EEGCalcs.sig, trial.EEGCalcs.fs)
