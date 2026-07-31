from types import ModuleType
from enum import Enum
import pandas as pd
import warnings
import numpy as np
from pathlib import Path
from ephysiopandas.data_io import load_trial, save_dataframe
from ephysiopandas.helpers import print_dataframe_insert_details, assert_columns_exist
from ephysiopandas.config import RecordingType


def insert_trial_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inserts a 'trial_type' column into the dataframe df based on
    the filename patterns provided in kws.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to insert the trial type into.
    kws : dict
        A dictionary where keys are trial type names and values
        are lists of substrings to search for in filenames.

    Returns
    -------
    pd.DataFrame
        The updated dataframe with the 'trial_type' column.
    """
    assert_columns_exist(df, ["filename"])

    if "trial_type" not in df.columns:
        df["trial_type"] = "unknown"

    for row in df.iterrows():
        fname = row[1].filename
        cluster = row[1].cluster
        channel = row[1].channel
        if Path(fname).is_dir():
            insert_measure_into_df(
                df, fname, cluster, channel, "OpenEphys", "trial_type"
            )
        else:
            insert_measure_into_df(df, fname, cluster, channel, "Axona", "trial_type")

    return df


@print_dataframe_insert_details
def insert_dict_into_df(
    df: pd.DataFrame,
    fname: Path,
    cluster: int | None,
    channel: int | None,
    measure_dict: dict,
    measure_name: str,
    run_dir: str | None = None,
) -> pd.DataFrame:
    """
    Inserts the values contained in measure_dict into the dataframe df
    at the row corresponding to the filename, cluster, and channel if present
    or match to filename if cluster and channel are None.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to insert the measure into.
    measure_dict : dict
        A dictionary where keys are tuples of (filename, cluster, channel)
        and values are the measure values to insert.
    measure_name : str
        The name of the measure to insert (i.e. the column name in df).
    run_dir: str | None
        The run direction to insert the measure for (e.g. "w" or "e
        for linear track trials). If None then the measure will be inserted
        without considering run direction. If not None then the measure will be
        inserted considering run direction and the column name will be
        measure_name + "_" + run_dir + "_" + key

    Notes
    -----
    If the name doesn't already exist in df then each key in measure_dict
    will be inserted as a column into df with the
    measure_name prepended to the key name and the value will be inserted
    into the corresponding row in df. Otherwise the value will be updated
    in the corresponding row.
    """
    for key in measure_dict.keys():
        col_name = measure_name + "_" + str(key)
        if col_name not in df.columns:
            df[col_name] = np.nan

    for key, measure in measure_dict.items():
        if isinstance(key, Enum):
            col_name = measure_name + "_" + str(key.value)
        else:
            col_name = measure_name + "_" + str(key)

        if cluster is not None and channel is not None:
            if run_dir is not None:
                assert_columns_exist(
                    df, ["filename", "cluster", "channel", "run_direction"]
                )
                index = df[
                    np.logical_and(
                        df.run_direction == run_dir,
                        np.logical_and(
                            np.logical_and(
                                df.cluster == cluster, df.channel == channel
                            ),
                            df.filename == fname,
                        ),
                    )
                ].index[0]

            else:
                assert_columns_exist(df, ["filename", "cluster", "channel"])
                index = df[
                    np.logical_and(
                        np.logical_and(df.cluster == cluster, df.channel == channel),
                        df.filename == fname,
                    )
                ].index[0]

        else:
            assert_columns_exist(df, ["filename"])
            index = df[df.filename == fname].index[0]
        df.at[index, col_name] = measure

    return df


@print_dataframe_insert_details
def insert_measure_into_df(
    df: pd.DataFrame,
    fname: Path,
    cluster: int | None,
    channel: int | None,
    measure: float,
    measure_name: str,
    run_dir: str | None = None,
) -> pd.DataFrame:
    """
    Inserts the value contained in measure into the dataframe df
    at the row corresponding to the filename, cluster, and channel.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to insert the measure into.
    fname : PosixPath
        The filename corresponding to the trial.
    cluster : int
    channel : int
    measure : float
    measure_name : str
    """
    if measure_name not in df.columns:
        df[measure_name] = np.nan

    if cluster is not None and channel is not None:
        if run_dir is not None:
            assert_columns_exist(
                df, ["filename", "cluster", "channel", "run_direction"]
            )
        else:
            assert_columns_exist(df, ["filename", "cluster", "channel"])

        index = df[
            np.logical_and(
                np.logical_and(df.cluster == cluster, df.channel == channel),
                df.filename == fname,
            )
        ].index[0]

        df.at[index, measure_name] = measure

    else:
        assert_columns_exist(df, ["filename"])
        index = df[df.filename == fname]
        df.loc[index.index, measure_name] = measure

    return df


def populate_cluster_level_metrics(
    df: pd.DataFrame, vars: list, mod: ModuleType, **kws
) -> pd.DataFrame:
    """
    Populates the dataframe df with cluster metrics passed in kws.
    The keys of kws are the measure names and the values are dictionaries
    where keys are tuples of (filename, cluster, tetrode) and values are
    the measure values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to populate.
    vars : list
        The list of measure names to populate.
        These are called as functions from cluster_metrics and so
        must be the names of functions in cluster_metrics e.g.
        "num_spikes", "overdispersion" etc
    mod : ModuleType
        The module to import the functions from. This should be a module that
        contains functions that take a trial, cluster, and channel as input and
        return a measure to be added to the dataframe. This allows the user to
        specify any module that contains cluster level metrics that they want to
        add to the dataframe.
    kws : dict
        This should be a dictionary of dictionaries where keys are
        measure names and values are dictionaries of kwargs to pass
        to the relevant measure
        append_to_name - a string to append to the measure name
        when inserting into the dataframe (see above)
    """

    assert_columns_exist(df, ["filename", "cluster", "channel"])

    for group in df.groupby("filename"):
        fname = Path(group[0])

        trial = load_trial(fname)

        for row in group[1].itertuples():
            cluster = int(row.cluster)
            channel = int(row.channel)

            for var in vars:
                func = getattr(mod, var)
                try:
                    measure = func(trial, cluster, channel, **kws)
                except Exception as e:
                    measure = np.nan
                    warnings.warn(
                        f"Error calculating {var} for {fname}, cluster {
                            cluster
                        }, channel {channel}: {e}"
                    )

                if kws.get("append_to_name", None):
                    var_name = var + "_" + kws["append_to_name"]
                else:
                    var_name = var

                if isinstance(measure, dict):
                    insert_dict_into_df(df, fname, cluster, channel, measure, var_name)
                else:
                    insert_measure_into_df(
                        df, fname, cluster, channel, measure, var_name
                    )

    if kws.get("save_as", None):
        save_dataframe(df, kws["save_as"])

    return df


def populate_linear_track_metrics(
    df: pd.DataFrame, vars: list, mod: ModuleType, **kws
) -> pd.DataFrame:
    """
    Populates the dataframe df with cluster metrics passed in kws.
    The keys of kws are the measure names and the values are dictionaries
    where keys are tuples of (filename, cluster, tetrode) and values are
    the measure values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to populate.
    vars : list
        The list of measure names to populate.
        These are called as functions from cluster_metrics and so
        must be the names of functions in cluster_metrics e.g.
        "num_spikes", "overdispersion" etc
    mod : ModuleType
        The module to import the functions from. This should be a module that
        contains functions that take a trial, cluster, and channel as input and
        return a measure to be added to the dataframe. This allows the user to
        specify any module that contains cluster level metrics that they want to
        add to the dataframe.
    kws : dict
        This should be a dictionary of dictionaries where keys are
        measure names and values are dictionaries of kwargs to pass
        to the relevant measure
        append_to_name - a string to append to the measure name
        when inserting into the dataframe (see above)
    """

    assert_columns_exist(df, ["filename", "cluster", "channel", "run_direction"])

    run_dirs = ["w", "e"]

    for group in df.groupby("filename"):
        fname = Path(group[0])

        trial = load_trial(fname)

        for row in group[1].itertuples():
            cluster = int(row.cluster)
            channel = int(row.channel)

            for run_dir in run_dirs:
                for var in vars:
                    func = getattr(mod, var)
                    try:
                        measure = func(trial, cluster, channel, run_dir, **kws)
                    except Exception as e:
                        measure = np.nan
                        warnings.warn(
                            f"Error calculating {var} for {fname}, cluster {
                                cluster
                            }, channel {channel}, run direction {run_dir}: {e}"
                        )

                    if kws.get("append_to_name", None):
                        var_name = var + "_" + kws["append_to_name"]
                    else:
                        var_name = var

                    if isinstance(measure, dict):
                        insert_dict_into_df(
                            df, fname, cluster, channel, run_dir, measure, var_name
                        )
                    else:
                        insert_measure_into_df(
                            df, fname, cluster, channel, run_dir, measure, var_name
                        )

    if kws.get("save_as", None):
        save_dataframe(df, kws["save_as"])

    return df


def populate_trial_level_metrics(df: pd.DataFrame, vars: list, mod: ModuleType, **kws):
    """
    A more general version of a method to populate a dataframe with trial level
    metrics.

    This allows the user to specify any module and any list of
    functions that can be used to populate any trial level metrics from any
    module (e.g. trial_metrics, lfp_metrics, or any user defined module).

    The user must specify the module to use as well as the list of variables
    to calculate and add to the dataframe.

    This is useful for allowing users to add their own custom trial level metrics
    without needing to modify the code in this file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to populate.
    vars : list
        The list of measure names to populate.
        These are called as functions from the module specified in mod and so
        must be the names of functions in that module e.g. "mean_firing_rate",
        "peak_firing_rate", "gamma_power" etc
    mod : ModuleType
        The module to import the functions from. This should be a module that
        contains functions that take a trial as input and return a measure to be
        added to the dataframe. This allows the user to specify any module that
        contains trial level metrics that they want to add to the dataframe.
    kws : dict
        This should be a dictionary of dictionaries where keys are
        measure names and values are dictionaries of kwargs to pass
        to the relevant measure. This allows the user to specify any keyword
        arguments to pass to the functions that calculate the trial level metrics.

    Returns
    -------
    df : pd.DataFrame
        The dataframe with the trial level metrics added.

    Notes
    -----
    The user might need to append some info to the variable name
    so that the right name is added to the dataframe when it's
    possible to call the function with keyword arguments that
    modify the underlying calculation e.g. when examining fast or
    slow gamma

    **kws - keyword arguments to pass to the trial metric functions.
            the keys are arguments to the functions and the values
            are the values to pass.

            append_to_name - a string to append to the measure name
            when inserting into the dataframe (see above)

    """
    assert_columns_exist(df, ["filename", "mouse_name"])

    for group in df.groupby("filename"):
        fname = Path(group[0])
        trial = load_trial(fname)

        for var in vars:
            func = getattr(mod, var)

            try:
                measure = func(trial, **kws)
            except Exception as e:
                measure = np.nan
                warnings.warn(f"Error calculating {var} for {fname}: {e}")

            if kws.get("append_to_name", None):
                var_name = var + "_" + kws["append_to_name"]
            else:
                var_name = var

            if isinstance(measure, dict):
                insert_dict_into_df(df, fname, None, None, measure, var_name)
            else:
                insert_measure_into_df(df, fname, None, None, measure, var_name)

    if kws.get("save_as", None):
        save_dataframe(df, kws["save_as"])

    return df


def add_cell_type(df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the cell type (Pyramidal or interneuron) to the
    dataframe

    Paramters
    ---------
    df : pd.DataFrame
        The dataframe to add the cell type to
    trial : str
        The type of trial the dataframe corresponds to
        (linear_track or open_field)

    Returns
    -------
    df : pd.DataFrame
        The dataframe with the cell type added
    """
    assert_columns_exist(df, ["mouse_name", "channel", "cluster"])
    assert_columns_exist(
        classification_df, ["mouse_name", "channel", "cluster", "cell_type"]
    )
    # Load the relevant cell classification dataframe
    if "cell_type" not in classification_df.columns:
        warnings.warn("cell_type column not found.")
        warnings.warn("Classify cells first (see cell_classification.py")

    df = df.merge(
        classification_df[["mouse_name", "channel", "cluster", "cell_type"]],
        on=["mouse_name", "channel", "cluster"],
    )

    return df


def add_recording_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the recording type (tetrode or neuropixel) to the
    dataframe based on the channel number. If the channel number is
    less than 32 then it is classified as a tetrode recording, otherwise
    it is classified as a neuropixel recording.

    Paramters
    ---------
    df : pd.DataFrame
        The dataframe to add the recording type to

    Returns
    -------
    df : pd.DataFrame
        The dataframe with the recording type added
    """

    df["recording_type"] = df.filename.apply(
        lambda x: RecordingType.OpenEphys if Path(x).is_dir() else RecordingType.Axona
    )

    return df


def add_KSLabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the trial is of type OpenEphys then adds the KSLabel
    to the dataframe based on the filename and the cluster identity.

    Notes
    -----
    Defaults to a 'good' label as the dataframe may contain Axona data
    and it's assumed this has been curated by hand and is therefore good.
    """

    assert_columns_exist(df, ["filename", "cluster"])

    df["KSLabel"] = "good"

    for group in df.groupby("filename"):
        fname = group[0]

        if Path(fname).is_dir():
            trial = load_trial(fname)
            if not trial.template_model:
                trial.load_neural_data()

            for row in group[1].itertuples():
                cluster = row.cluster
                try:
                    df["KSLabel"] = trial.template_model.metadata["KSLabel"][cluster]
                except Exception:
                    df["KSLabel"] = "bad"

    return df


def remove_no_spike_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows from the dataframe where the cluster has no spikes.
    This is determined by the presence of a column called "num_spikes"
    and the value in that column being 0.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to remove no spike clusters from

    Returns
    -------
    df : pd.DataFrame
        The dataframe with no spike clusters removed
    """
    if "num_spikes" not in df.columns:
        warnings.warn("num_spikes column not found. Cannot remove no spike clusters.")
        return df

    df = df[df.num_spikes > 0]

    return df


def remove_redundant_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    When merging/ joining dataframes sometimes duplicated columns
    are added as I guess pandas doesn't know how to resolve when
    two dataframes have the same name (I don't care to find out
    what I'm doing wrong). This function removes the duplicates as
    they are added in as name_x or name_y
    """
    if name + "_x" in df.columns:
        df = df.rename(columns={name + "_x": name})
    if name + "_y" in df.columns:
        df = df.rename(columns={name + "_y": name})

    df = df.loc[:, ~df.columns.duplicated()]

    return df


def convert_pkls(directory: Path, convert_to: str = "csv"):
    """
    Converts all the pandas dataframes saved as pkls in the specified
    directory to the specified format (csv or excel).
    """
    assert convert_to in ["csv", "excel"], "convert_to must be 'csv' or 'excel'"

    pkl_files = list(directory.glob("*.pkl"))

    for pkl_file in pkl_files:
        df = pd.read_pickle(pkl_file)
        if convert_to == "csv":
            out_file = pkl_file.with_suffix(".csv")
            df.to_csv(out_file, index=False)
        elif convert_to == "excel":
            out_file = pkl_file.with_suffix(".xlsx")
            df.to_excel(out_file, index=False)


def tetrode_cluster_dict_per_file(df: pd.DataFrame, fname: Path) -> dict:
    """
    Returns a dictionary where keys are tetrodes and items are
    clusters for the given filename

    """
    assert_columns_exist(df, ["filename", "tetrode", "cluster"])

    assert fname in set(df.filename)

    file_df = df[df.filename == fname]

    out = {}
    for tetrode, group in file_df.groupby("tetrode"):
        out[tetrode] = group.cluster.to_list()

    return out


def tetrode_cluster_dict_per_mouse(df: pd.DataFrame, mouse: str) -> dict:
    """
    Returns a dictionary where keys are tetrodes and items are
    clusters for the given mouse

    """
    assert_columns_exist(df, ["filename", "tetrode", "cluster", "mouse_name"])

    assert mouse in set(df.mouse_name)

    mouse_df = df[df.mouse_name == mouse]

    out = {}
    for tetrode, group in mouse_df.groupby("tetrode"):
        out[tetrode] = group.cluster.to_list()

    return out
