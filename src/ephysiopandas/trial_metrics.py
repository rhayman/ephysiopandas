import pandas as pd
import numpy as np
from ephysiopy.io.recording import TrialInterface as Trial
from ephysiopy.common.fieldcalcs import (
    thigmotaxis_score,
)
from ephysiopandas.config import RecordingType


def set_recording_type(row: pd.Series) -> RecordingType:
    """
    Sets the recording type for a given row in a dataframe.
    Parameters
    ----------
    row : pd.Series
        A row from a pandas DataFrame containing recording metadata.
    Returns
    -------
    RecordingType
        The recording type determined from the row data.
    """
    if "Trial" in row:
        if ".set" in row["Trial"]:
            return RecordingType.Axona
        else:
            return RecordingType.OpenEphys
    elif "filename" in row:
        if ".set" in row["filename"]:
            return RecordingType.Axona
        else:
            return RecordingType.OpenEphys
    else:
        raise ValueError("Row must contain either 'Trial' or 'filename' column.")


def thigmotaxis(trial: Trial, **kws) -> float:
    """
    Calculates the thigmotaxis score for a given trial.

    Parameters
    ----------
    trial : TrialInterface
        The trial object containing position data.

    Notes
    -----
    thigmo_score (float) - a score which is the ratio of the time spent
    in the inner portion of an environment to the time spent in the outer
    portion. The portions are allocated so that they have equal area.

    """

    xy = trial.PosCalcs.xy
    return thigmotaxis_score(xy)


def distance_traversed(trial: Trial, **kws) -> float:
    """
    Calculates the total distance traveled during a trial.

    Parameters
    ----------
    trial : TrialInterface
        The trial object containing position data.

    Returns
    -------
    float
        The total distance traveled in the trial.
    """
    xy = trial.PosCalcs.xy
    xy = np.ma.masked_invalid(xy)
    diffs = np.ma.diff(xy, append=xy[:, -1].reshape(-1, 1))
    return np.ma.sum(np.ma.abs(np.hypot(diffs[0], diffs[1])))


def mean_speed(trial: Trial, **kws) -> float:
    """
    Calculates the mean speed during a trial.

    Parameters
    ----------
    trial : TrialInterface
        The trial object containing position data.

    Returns
    -------
    float
        The mean speed in the trial.
    """
    speed = trial.PosCalcs.speed
    speed = np.ma.masked_invalid(speed)
    return np.ma.mean(speed)
