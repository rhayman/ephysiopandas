"""
Functions for generating shuffled distributions for a collection of metrics
"""

from types import ModuleType
from pathlib import Path
import pandas as pd
import numpy as np
from ephysiopy.common.utils import flatten_list
from ephysiopy.phase_precession.linear_track import run_phase_analysis
from ephysiopy.phase_precession.phase_precession import shuffle_regression
import ephysiopandas.data_io as data_io
from ephysiopandas.helpers import assert_columns_exist


def shuffle_metrics(
    df: pd.DataFrame, metrics: list, mod: ModuleType, save_dir: Path, **kws
):
    """
    Valid measures to shuffle:
        spatial_info - generate ratemaps
        coherence - generate ratemaps
        grid_score - generate ratemaps
        correlation measures - generate ratemaps
        phase measures wrt theta if spike train shuffle
    """
    n_shuffles = kws.get("n_shuffles", 500)
    map_binsize = kws.get("map_binsize", 3)

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # iterate over the metrics and then the dataframe rows
    for metric in metrics:
        shuffled_measures = []
        for _, row in df.iterrows():
            # generate the shuffled distribution for the metric
            # save the distribution to a file
            trial = data_io.load_trial(row.filename)
            cluster = row.cluster
            channel = row.channel
            # store the scores in a list before flattening as saving
            func = getattr(mod, metric)
            scores = func(
                trial,
                cluster,
                channel,
                do_shuffle=True,
                n_shuffles=n_shuffles,
                map_binsize=map_binsize,
                **kws,
            )
            shuffled_measures.append(scores)
        shuffled_measures = flatten_list(shuffled_measures)
        np.save(
            save_dir / Path(metric + "_shuffle.npy"),
            np.array(shuffled_measures),
        )


def shuffle_phase_precession_regressions(
    df: pd.DataFrame, n_shuffles: int = 10000, save_dir: Path = Path.cwd(), **kws
) -> None:
    """
    Performs shuffles to generate a null distribution for phase precession regression metrics.

    Parameters:
    -----------
    n_shuffles : int
        The **total** number of shuffles to perform.
        Default is 10000.
    save_dir : Path
        The directory where the shuffled distributions will be saved.

    """
    assert_columns_exist(df, ["filename", "cluster", "channel", "run_direction"])

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    shuffles_per_trial = n_shuffles // len(df)

    all_slopes = []

    for _, row in df.iterrows():
        # generate the shuffled distribution for the metric
        # save the distribution to a file
        trial = data_io.load_trial(row.filename)
        cluster = row.cluster
        channel = row.channel
        run_dir = row.run_direction
        pp_results, _ = run_phase_analysis(
            trial,
            cluster,
            channel,
            run_direction=run_dir,
            return_pp=True,
            **kws,
        )

        sr = [shuffle_regression(p, nshuffles=shuffles_per_trial) for p in pp_results]
        sr = flatten_list(sr)
        # s is a list of RegressionResults, so extract out the slope/
        # correlation coefficient
        slopes = [pp.stats.slope for pp in sr]
        all_slopes.append(slopes)

    # save to file
    all_slopes = flatten_list(all_slopes)
    np.save(save_dir / Path("phase_precession_slope_shuffle.npy"), np.array(all_slopes))
