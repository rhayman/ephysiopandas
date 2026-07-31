import numpy as np
from datetime import datetime
import pandas as pd
import functools
from pathlib import Path
from skimage.measure import points_in_poly, CircleModel
from ephysiopy.io.bases import TrialInterface as Trial
from ephysiopy.common.fieldproperties import FieldProps, fieldprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from ephysiopy.common.fieldcalcs import (
    fancy_partition,
    simple_partition,
)
from ephysiopandas.data_io import load_trial


def print_dataframe_insert_details(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fname = args[1]
        cluster = args[2]
        channel = args[3]
        measure = args[5]
        print(
            f"Inserting {measure} for {Path(fname).stem}, cluster {cluster}, channel {
                channel
            }"
        )
        return func(*args, **kwargs)

    return wrapper


# prints out function name, trial pathname, cluster and channel id
def print_measure_details(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        trial = args[0]
        cluster = args[1]
        channel = args[2]
        print(
            f"Calculating {func.__name__} for {trial.pname}, cluster {
                cluster
            }, channel {channel}"
        )
        return func(*args, **kwargs)

    return wrapper


def print_row_details(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        row = args[0]
        print(
            f"Calculating {func.__name__} for {Path(row.filename).stem}, cluster {
                row.cluster
            }, channel {row.channel}"
        )
        return func(*args, **kwargs)

    return wrapper


def assert_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Asserts that all columns in the list 'columns' exist in the dataframe 'df'.
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    columns : list[str]
        The list of column names to check for.
    """
    for col in columns:
        assert col in df.columns, f"Column '{col}' not found in dataframe."


def single_or_list(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    return wrapper


def get_environment_shape(xy: np.ndarray) -> tuple:
    """
    Given an array of x,y coordinates, fits a circular model
    to the points and returns "circle" if the points fit well within
    the circle (95% of points inside), otherwise returns "square".

    Parameters
    ----------
    xy : np.ndarray
        An Nx2 array of x,y coordinates.

    Returns
    -------
    np.ndarray
        An Mx2 array of x,y coordinates representing the
        perimeter of the fitted circle.
    """
    xy = np.ma.masked_where(np.isnan(xy), xy)
    centre = np.ma.ptp(xy, axis=0) / 2 + np.ma.min(xy, axis=0)
    radius = np.ma.mean(np.ma.ptp(xy, axis=0)) / 2
    # Slightly increase radius to ensure all points are inside
    # if circular environment
    radius *= 1.05
    model = CircleModel(centre, radius)
    t = np.linspace(0, 2 * np.pi, 101)
    perimeter_points = model.predict_xy(t)

    inside = points_in_poly(xy, perimeter_points)

    if np.sum(inside) / len(xy) >= 0.95:
        return "circle", perimeter_points
    else:
        return "square", perimeter_points


def get_environment_type(xy: np.ndarray) -> str:
    """
    Given an array of x,y coordinates, check the ratio of
    the peak to peak x y range to see if this is a linear
    track or not.

    Parameters
    ----------
    xy : np.ndarray
        An Nx2 array of x,y coordinates.

    Returns
    -------
    str
        'linear' if linear track (ratio < 0.2) or '2D' if not
    """

    a, b = np.ptp(xy, axis=-1)

    a = a if a > b else b

    ratio = b / a

    if ratio < 0.2:
        return "linear"
    else:
        return "2D"


def get_date_from_trial(trial: Trial) -> datetime:
    """
    Extracts the date from the trial's pathname.

    Parameters
    ----------
    trial : Trial
        The trial from which to extract the date.

    Returns
    -------
    datetime.datetime
        The extracted date.

    Example
    -------
    Probably a faster way to do this with a groupby first...
    >>> from ephysiopandas.data_io import load_trial
    >>> df['date'] = df.apply(lambda row: get_date_from_trial(load_trial(row)))
    """

    dt = datetime.now()

    if trial.pname.is_dir():  # OpenEphys
        for elem in trial.settings.tree.iter("INFO"):
            for d in elem.iter("DATE"):
                date_str = d.text
                dt = datetime.strptime(date_str, "%d %b %Y %H:%M:%S")

    elif trial.pname.is_file():  # Axona
        date_str = trial.settings["trial_date"]
        dt = datetime.strptime(date_str, "%A, %d %b %Y")

    return dt


def get_equal_bin_edges(
    t1_pname: Path, t2_pname: Path, ppm=300, map_binsize=3
) -> list[np.ndarray]:
    """
    Calculate equal bin edges for two trials

    Parameters
    ----------
    trial1 (AxonaTrial) - the first trial
    trial2 (AxonaTrial) - the second trial

    Returns
    -------
    list[np.ndarray] - the np.ndarray for the bin edges
    """

    t1 = load_trial(t1_pname, MAP_BINSIZE=map_binsize)
    t2 = load_trial(t2_pname, MAP_BINSIZE=map_binsize)

    xlims1 = t1.RateMap._x_lims
    xlims2 = t2.RateMap._x_lims
    ylims1 = t1.RateMap._y_lims
    ylims2 = t2.RateMap._y_lims

    xl = (min(xlims1[0], xlims2[0]), max(xlims1[1], xlims2[1]))
    yl = (min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1]))
    t1.RateMap._x_lims = xl
    t1.RateMap._y_lims = yl
    t1.RateMap.var2Bin = t1.RateMap.var2Bin

    be = t1.RateMap._calc_bin_edges(map_binsize)

    return be


def get_field_props(
    trial: Trial,
    cluster: int,
    channel: int,
    partition_method="fancy",
    field_rate_thresh=50,
    min_run_len=50,
    min_field_sz_bins=3,
    exclude_speeds=[0, 0.5],
    map_binsize=3,
    env="2D",
    **kws,
) -> list[FieldProps]:
    """
    Gets the list of FieldProps for the cluster.

    Some of the arguments that can be provided only kick in
    for the fancy_partition or simple_partition methods. More
    or less you want to use the fancy_partition method of field
    finding for 2D / open field and the simple_partition method
    for linear track data

    Parameters
    ----------
    trial - Trial instance
    cluster - int
    channel - int
    partition_method - str
        either 'fancy' or 'simple'
    field_rate_thresh - int
    min_run_len - int
    min_field_sz_bins - int
    exclude_speeds - list
    map_binsize - int
    env - str
        either '2D' or 'linear'

    """
    ppm = int(trial.settings["tracker_pixels_per_metre"])

    trial.PosCalcs.ppm = ppm  # triggers postprocesspos()

    # partition the cells firing into distinct fields
    if env == "linear":
        binned_data = trial.get_linear_rate_map(
            cluster, channel, binsize=map_binsize, **kws
        )
    else:
        binned_data = trial.get_rate_map(
            cluster, channel, binsize=map_binsize, **kws)
    # get the field properties
    if "fancy" in partition_method:
        _, _, label_image, _ = fancy_partition(
            binned_data,
            field_threshold_percent=field_rate_thresh,
        )
    else:
        _, _, label_image, _ = simple_partition(
            binned_data,
            rate_threshold_prc=field_rate_thresh,
        )
    # Filter out small fields from the image partition here...
    label_image = remove_small_objects(label_image, max_size=min_field_sz_bins)
    # and relabel the label image sequentially
    label_image, _, _ = relabel_sequential(label_image)

    pos_data = trial.PosCalcs.xy
    spike_times = trial.get_spike_times(cluster, channel)

    field_props = fieldprops(
        label_image,
        binned_data,
        spike_times,
        pos_data,
        min_run_length=min_run_len,
    )

    return field_props
