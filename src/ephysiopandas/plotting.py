import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import pandas as pd
from pathlib import Path
from skimage import measure
import ephysiopandas.data_io as data_io
from ephysiopy.common.waveformcalcs import peak_local_max_time
from ephysiopandas.helpers import (
    get_field_props,
    print_row_details,
)


@print_row_details
def plot_open_field_cluster_summary(
    row: pd.Series,
    save: bool = False,
    **kws,
):
    """
    Plots a summary figure for the provided cluster and channel
    including path, rate map, autocorr, phase of spiking etc

    Parameters
    ----------
    row : pd.Series
        A row from the cluster metrics dataframe containing the necessary
        information to load the trial and plot the relevant metrics.
    save : bool, optional
        Whether to save the figure instead of showing it. Default is False.
    save_dir : Path, optional
        The directory where to save the figure if save is True. Default is the
        current directory.

    """
    # if asked to save make sure we have a save directory
    if save:
        if "save_dir" not in kws:
            print(
                "Warning: save is True but no save_dir provided. Saving to current directory."
            )
        save_dir = kws.get("save_dir", Path.cwd())

    output_format = kws.get("figure_format", "svg")

    trial = data_io.load_trial(row.filename)
    cluster = int(row.cluster)
    try:
        channel = int(row.channel)
    except AttributeError:
        channel = int(row.tetrode)

    try:
        cell_type = row.cell_type
    except Exception:
        cell_type = "Unknown"

    fig = plt.figure(figsize=(16, 10))

    title = f"{trial.pname.stem} - Cluster {cluster} Tetrode {channel}\n{cell_type}"

    print(title)

    fig.suptitle(title, fontweight="bold", fontsize=18)

    ax1 = plt.subplot(321)
    ax2 = plt.subplot(323)
    ax3 = plt.subplot(325)
    ax4 = plt.subplot(322)
    ax5 = plt.subplot(324)
    ax6 = plt.subplot(326, projection="polar")

    fs = 14

    # ) Autocorrelations at 2 different window sizes
    trial.plot_acorr(
        cluster,
        channel,
        binsize=0.001,
        Trange=[-0.02, 0.02],
        ax=ax1,
        **kws,
    )
    ax1.set_xticks([-0.02, 0, 0.02])
    ax1.set_xticklabels(["-20ms", "0", "20ms"])
    c = "$\\bf{Contamination}$"
    Q = "$\\bf{Q}$"
    R = "$\\bf{R}$"
    # add info about contamination of ISI (used by KiloSort)
    ax1.text(
        -0.35,
        0.5,
        f"{c}\n{Q}: {row.contamination_Q:.4f}\n{R}: {row.contamination_R:.4f}\n",
        transform=ax1.transAxes,
        fontsize=fs,
        verticalalignment="center",
    )
    trial.plot_acorr(
        cluster,
        channel,
        binsize=0.005,
        ax=ax2,
        **kws,
    )
    ax2.set_xticks([-0.5, 0, 0.5])
    ax2.set_xticklabels(["-500ms", "0", "500ms"])
    # add info about theta modulation metrics
    t = "$\\bf{Theta\\ modulation}$"
    acorr_str = f"{t}\n{row.theta_mod_v3:.2f}\n"
    ax2.text(
        -0.35,
        0.5,
        acorr_str,
        transform=ax2.transAxes,
        fontsize=fs,
        verticalalignment="center",
    )

    # 2) Cluster waveforms
    trial.plot_waveforms(
        cluster,
        channel,
        ax=ax3,
        axes_labels=True,
        **kws,
    )
    # annotate with lines showing spike width and AHP decay time
    S = data_io.get_WaveformCalcs(trial, cluster, channel)
    waveform, _ = S.mean_waveform(S.get_best_channel())
    n_samples = waveform.shape[0]
    if waveform.shape[0] == 50:
        _fs = 5e4  # 50 samples at 50 kHz
        time = np.linspace(-200, 800, n_samples)  # in microseconds
    else:
        _fs = 3e4  # 80 samples at 30 kHz
        time = np.linspace(-2000, 2000, n_samples)  # in microseconds

    p2t = peak_local_max_time(waveform, fs=_fs)
    if not np.isnan(p2t["trough_idx"]):
        for vl in [p2t["trough_idx"], p2t["peak_idx"]]:
            ax3.axvline(
                time[vl],
                color="r",
                linestyle="--",
            )
    # add info about ahp_decay time and spike width
    sw = "$\\bf{Spike\\ width}$"
    ahp = "$\\bf{AHP\\ decay}$"
    ax3.text(
        0.5,
        -0.1,
        f"{sw}: {row.spike_width:.2f}μs\n{ahp}: {row.ahp_decay:.2f}μs",
        transform=ax3.transAxes,
        fontsize=fs,
        verticalalignment="top",
        ha="center",
    )

    # 3) Path + spikes
    trial.plot_spike_path(
        cluster,
        channel,
        ms=1.5,
        equal_axes=True,
        ax=ax4,
        **kws,
    )
    # add some annotations about spikes, spatial info etc
    ax4.text(
        1.05,
        0.5,
        f"$\\bf{{N\\ spikes}}$: {row.num_spikes}\n",
        transform=ax4.transAxes,
        fontsize=fs,
        verticalalignment="center",
    )
    # Plot the rate map - check if linear track or open field
    # and plot appropriately - default to open field

    try:
        if row.environment:
            env_type = row.environment
        else:
            env_type = "open_field"
    except Exception:
        env_type = "open_field"

    # set the partition method for field detection as well
    if env_type == "linear_track":
        trial.plot_linear_rate_map(cluster, channel, ax=ax5, **kws)
        if "partition_method" not in kws.keys():
            kws["partition_method"] = "simple"
        env = "linear"
        min_run_len = kws.pop("min_runs", 25)
    else:
        trial.plot_rate_map(cluster, channel, ax=ax5, **kws)
        if "partition_method" not in kws.keys():
            kws["partition_method"] = "fancy"
        env = "2D"
        min_run_len = kws.pop("min_runs", 5)

    # annotate with spatial info, overdispersion, sparsity etc
    s = "$\\bf{Field\\ size}$"
    mr = "$\\bf{Mean\\ rate}$"
    pr = "$\\bf{Peak\\ rate}$"
    c = "$\\bf{Coherence}$"
    od = "$\\bf{Overdispersion}$"
    si = "$\\bf{Spatial\\ info}$"
    ax5.text(
        1.05,
        0.5,
        f"{s}: {row.field_size: .2f} cm²\n"
        f"{mr}: {row.mean_rate: .2f} Hz\n"
        f"{pr}: {row.peak_rate:.2f} Hz\n"
        f"{c}: {row.coherence:.2f}\n"
        f"{od}: {row.overdispersion:.2f}\n"
        f"{si}: {row.spatial_info:.2f} bits/spike",
        transform=ax5.transAxes,
        fontsize=fs,
        verticalalignment="center",
    )
    # add the field contours from the get_field_props function in
    # ephysiopandas
    if not np.isnan(row.field_size):
        try:
            fp = get_field_props(
                trial,
                cluster,
                channel,
                field_rate_thresh=200,
                min_run_len=min_run_len,
                env=env,
                **kws,
            )
            if fp:
                # fp = sort_fields_by_attr(fp, "area")
                bd = fp[0].binned_data
                yedges = bd.bin_edges[0]
                if len(bd.bin_edges) > 1:
                    xedges = bd.bin_edges[1]
                for field_prop in fp:
                    contour = measure.find_contours(field_prop.label_image)
                    for c in contour:
                        ix = np.ceil(c[:, 1]).astype(int)
                        iy = np.ceil(c[:, 0]).astype(int)
                        ax5.plot(xedges[ix], yedges[iy], color="k", linewidth=3)
        except Exception:
            pass

    trial.plot_hd_map(cluster, channel, ax=ax6, add_mrv=True, **kws)
    # annotate with kl div score, mrv length and angle
    mrl = "$\\bf{MRV\\ length}$"
    mra = "$\\bf{MRV\\ angle}$"
    ax6.text(
        1.05,
        0.5,
        f"{mrl}: {row.head_dir_mean_resultant_length:.2f}\n{mra}: {
            np.rad2deg(row.head_dir_mean_resultant_angle):.2f}°",
        transform=ax6.transAxes,
        fontsize=fs,
        verticalalignment="center",
    )

    if save:
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        fname = Path(
            f"{trial.pname.stem}_cluster_{cluster}_channel_{channel}.{output_format}"
        )

        plt.savefig(save_dir / fname, format=output_format, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()

    return row


def plot_bargraph(
    data: pd.DataFrame,
    y: str,
    x: str,
    hue: str | None = None,
    **kws,
):
    """
    Plots a bar graph of the provided measure grouped by the provided
    groupby column

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the data to plot
    measure : str
        The column name of the measure to plot
    groupby : str
        The column name to group the data by (Phenotype, etc)
    Mann-Whitney
    """

    ax = kws.get("ax", None)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    if hue is None:
        ax = sns.barplot(x=x, y=y, data=data, errorbar="se", ax=ax)
    else:
        ax = sns.barplot(x=x, y=y, hue=hue, data=data, errorbar="se", ax=ax)
    pair = data[x].unique().tolist()
    annot = Annotator(ax, [pair], data=data, x=x, y=y)
    annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
    annot.apply_test()
    annot.annotate()
    ax.set_title(f"{y}")
    return ax


def plot_spike_path_ratemap(
    row: pd.Series,
    **kws,
):
    """
    Plots a summary figure for the provided cluster and channel
    including path, rate map, autocorr, phase of spiking etc

    Parameters
    ----------
    row : pd.Series
        A row from the cluster metrics dataframe containing the necessary
        information to load the trial and plot the relevant metrics.
    """

    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    trial = data_io.load_trial(row.filename)
    cluster = int(row.cluster)
    channel = int(row.channel)
    # 3) Path + spikes
    trial.plot_spike_path(
        cluster,
        channel,
        ms=1.5,
        equal_axes=True,
        ax=ax[0],
        **kws,
    )

    trial.plot_rate_map(
        cluster,
        channel,
        equal_axes=True,
        ax=ax[1],
        **kws,
    )
    return ax


def plot_deciles(
    df: pd.DataFrame,
    measure: str,
    groupby: str,
    trial_type: str = "open_field",
    **kws,
):
    """
    Plots a bar graph of the provided measure grouped by the provided
    groupby column

    Also plots example ratemaps and spike /path plots of exemplars from
    each decile

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data to plot
    measure : str
        The column name of the measure to plot
    groupby : str
        The column name to group the data by (Phenotype, etc)
    trial_type : str
        The type of trial to plot (open_field or linear_track)
    """
    df = df[df.environment == trial_type].reset_index(drop=True)
    # calculate deciles for each group
    deciles = df.groupby(groupby)[measure].quantile(np.linspace(0, 0.9, 10)).unstack()

    deciles = deciles.reset_index().melt(
        id_vars=groupby, var_name="decile", value_name=measure
    )
    deciles.decile = deciles.decile.round(3)

    ax = kws.get("ax", None)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    sns.barplot(x="decile", y=measure, hue=groupby, data=deciles, errorbar="se", ax=ax)
    ax.set_title(f"{measure.capitalize().replace('_', ' ')} by {groupby} deciles")
    ax.set_ylabel(f"{measure.capitalize().replace('_', ' ')}")
    ax.set_xlabel("")

    # save the barplot
    save_dir = Path.cwd()

    if "save_dir" in kws:
        save_dir = kws["save_dir"]
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

    if save_dir is not None:
        fname = Path(f"{measure}_by_{groupby}_deciles.svg")
        plt.savefig(save_dir / fname, format="svg", bbox_inches="tight")
        plt.close("all")

    # save the deciles dataframe to csv
    deciles.to_csv(save_dir / Path(f"{measure}_by_{groupby}_deciles.csv"), index=False)

    # Plot example ratemaps and spike/path plots for exemplars from each decile
    # and for each group

    for group in df.groupby(groupby):
        for decile in deciles["decile"].unique():
            group_decile_df = group[1][
                (group[1][measure] > decile) & (group[1][measure] < decile + 0.1)
            ]
            if not group_decile_df.empty:
                # Get exemplar from this group/decile
                exemplar = group_decile_df.iloc[0]
                axs = plot_spike_path_ratemap(exemplar, save=False, **kws)
                # annotate the figure with group and decile info
                fig = axs[0].get_figure()
                fig.suptitle(
                    f"{group[0]} - {decile} decile : {exemplar.spatial_info:.2f} bits/spike",
                    fontsize=16,
                )
                save_name = Path(f"{group[0]}_{decile}_decile_{measure}.svg")
                fig.savefig(save_dir / save_name, format="svg", bbox_inches="tight")
                plt.close(fig)

    return ax
