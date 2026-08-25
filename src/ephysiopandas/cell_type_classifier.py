"""
Need to be careful when using Axona AND OpenEphys data in the same analysis
as the highpass filtering of the single unit data is likely different
Axona uses a bandpass filter with a low cutoff of 360 Hz and a high cutoff of 7 kHz, while
OpenEphys uses a highpass filter with unknown quantities because the documentation
is fucking awful
"""

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from pycircstat2.descriptive import circ_mean
from collections import OrderedDict
import ephysiopandas.data_io as data_io


# Extract the mean waveform for PCA
def get_mean_waveform(df: pd.DataFrame, **kws) -> pd.DataFrame:
    """
    Extracts mean waveform for the best channel for each unit in the dataframe.
    Preferentially loads and saves to disk as this will take some time...

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the spike data

    Returns
    -------
    pd.DataFrame
        Array of mean waveforms for each unit with same index as df
    """
    data_dir = kws.get("data_dir", None)

    if data_dir:
        # Check if mean waveforms are already saved to disk
        save_path = data_dir / "mean_waveforms_dataframe.hdf"
        if save_path.exists():
            print("Loading mean waveforms from disk...")
            return pd.read_hdf(save_path)

    # ensure the dataframe is sorted by index so that the mean waveforms are in
    # the same order as the dataframe
    # df.sort_index(inplace=True)
    # Assuming 50 samples per waveform
    mean_waveforms = np.empty((len(df), 50))

    i = 0
    for row in df.itertuples():
        trial = data_io.load_trial(row.filename, **kws)
        w = data_io.extract_mean_waveform(
            trial, int(row.cluster), int(row.channel), **kws
        )
        w = np.ravel(w)
        if w.shape[0] != 50:
            breakpoint()
        mean_waveforms[i, :] = w
        i += 1

    wave_df = pd.DataFrame(mean_waveforms, index=df.index)

    # save mean waveforms to disk for future use
    if data_dir:
        save_path = data_dir / "mean_waveforms_dataframe.hdf"
        print("Saving mean waveforms to disk...")
        wave_df.to_hdf(save_path, key="mean_waveforms", mode="w")

    return wave_df


def run_pca(waveforms: np.ndarray, **kws):
    """
    Classify cell type based on the mean waveforms of each cluster
    on it's best channel (highest power) using PCA.

    Parameters
    ----------
    waveforms : np.ndarray
        the input data where nrows is num samples and ncols is the
        number of features (i.e. each row is equal to a mean waveform)
    """
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0))

    embedded = pca.fit_transform(waveforms)


# Functions for classifying cells based on spike width and plotting
# the distribution of spike widths for the two cell types
def get_labels(spike_widths: np.ndarray):
    """
    Classify cells based on spike width using the k-means clustering
    method. Returns a list of labels for each spike width.
    """
    # whiten the spike widths to normalize the data
    # it's possible that spike widths were impossible to
    # determine due to a poorly formed waveform so replace
    # NaNs with a large spike width value
    idx = np.isfinite(spike_widths)
    spike_widths[~idx] = 1
    spike_widths = whiten(spike_widths.reshape(-1, 1))
    # Perform k-means clustering on spike widths
    rng = np.random.default_rng(42)
    km = kmeans2(spike_widths, 2, minit="random", rng=rng)
    labels = np.array(km[1])
    labels = np.array(km[1])

    # replace the integer labels with meaningful names
    # depending on which is larger
    cluster0_mean = np.mean(spike_widths[labels == 0])
    cluster1_mean = np.mean(spike_widths[labels == 1])
    if cluster0_mean < cluster1_mean:
        labels = np.where(labels == 0, "Interneuron", "Pyramidal")
    else:
        labels = np.where(labels == 0, "Pyramidal", "Interneuron")

    return labels


def get_decision_boundary(spike_widths: np.ndarray, labels: np.ndarray):
    """
    Calculate the decision boundary for the two clusters in labels
    given the spike widths

    Parameters
    ----------
    spike_widths : np.ndarray

    labels : np.ndarray

    Returns
    -------
    The decision boundary and the support vector classifier (SVC) model
    """
    from sklearn.svm import LinearSVC

    # it's possible that spike widths were impossible to
    # determine due to a poorly formed waveform so replace
    # NaNs with a large spike width value
    if spike_widths.ndim == 1:
        spike_widths = np.reshape(spike_widths, (-1, 1))

    sws = np.copy(spike_widths)
    idx = np.isfinite(sws)
    sws[~idx] = 1

    svc = LinearSVC(C=1000.0)
    svc.fit(sws, labels)
    decision_boundary = np.abs(svc.intercept_[0] / svc.coef_[0][0])

    return decision_boundary, svc


def plot_spike_width_dist(
    df: pd.DataFrame, decision_boundary: float, **kws
) -> plt.Figure:
    """
    Plot the distribution of spike widths for the two cell types
    in df. The dataframe should have a column "spike_width" and
    "cell_type".
    """
    assert "cell_type" in df.columns
    assert "spike_width" in df.columns

    bins = kws.get("bins", 20)
    annotate_font_size = kws.get("annotate_fs", 10)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # calculate means to annotate the plot
    int_mean = np.mean(df[df["cell_type"] == "Interneuron"]["spike_width"])
    int_std = np.std(df[df["cell_type"] == "Interneuron"]["spike_width"])
    pyr_mean = np.mean(df[df["cell_type"] == "Pyramidal"]["spike_width"])
    pyr_std = np.std(df[df["cell_type"] == "Pyramidal"]["spike_width"])

    # sort by value for plotting
    df.sort_values(by="cell_type", inplace=True, ascending=False)
    palette = kws.get("CELL_TYPE_COLOURS", sns.color_palette("tab10"))

    def __do_plot__(ax):

        sns.histplot(
            data=df,
            x="spike_width",
            hue="cell_type",
            stat="probability",
            bins=bins,
            element="step",
            kde=True,
            palette=palette,
            kde_kws={"cut": 3},
            ax=ax,
        )
        # plt.title("Cell classification based on spike width")
        plt.xlabel(r"Spike width ($\mu$s)")
        ax.set_xlim(0, 1000)
        int_col = palette[1]
        plt.axvline(int_mean, color=int_col, linestyle="--")
        textstr = "\n".join(
            (r"$\bar{x}=%.2f$$\mu$s" % (pyr_mean,), r"$\sigma=%.2f$$\mu$s" % (pyr_std,))
        )
        ax.annotate(
            textstr,
            xy=(pyr_mean, 0.8),
            xytext=(pyr_mean - 60, 1.01),
            textcoords=(ax.transData, ax.transAxes),
            xycoords=(ax.transData, ax.transAxes),
            color="k",
            fontsize=annotate_font_size,
            ha="left",
        )
        textstr = "\n".join(
            (r"$\bar{x}=%.2f$$\mu$s" % (int_mean,), r"$\sigma=%.2f$$\mu$s" % (int_std,))
        )
        ax.annotate(
            textstr,
            xy=(int_mean, 0.8),
            xytext=(int_mean - 60, 1.01),
            textcoords=(ax.transData, ax.transAxes),
            xycoords=(ax.transData, ax.transAxes),
            color="k",
            fontsize=annotate_font_size,
            ha="left",
        )
        textstr = rf"{decision_boundary:.2f}$\mu$s"
        ax.annotate(
            textstr,
            xy=(decision_boundary, 0.8),
            xytext=(decision_boundary - 40, 0.81),
            textcoords=(ax.transData, ax.transAxes),
            xycoords=(ax.transData, ax.transAxes),
            color="k",
            fontweight="bold",
            fontsize=annotate_font_size,
            ha="left",
        )

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        pyr_col = palette[0]
        plt.axvline(pyr_mean, color=pyr_col, linestyle="--")
        plt.axvline(decision_boundary, ymax=0.8, color="k", linestyle="--")
        p = mpatches.Patch(color=pyr_col, label="Pyramidal", alpha=0.4)
        p0 = mpatches.Patch(color=int_col, label="Interneuron", alpha=0.4)
        ax.legend(handles=[p, p0], loc="upper right", frameon=False)

    fig, ax = plt.subplots()
    __do_plot__(ax)

    return fig


# Functions for classifying the phase of theta at which cells fire


def get_theta_labels(theta_phases: np.ndarray):
    """
    Classify cells based on the theta phase at which they fire using
    k-means clustering. Returns a list of labels for each theta phase.

    Notes
    -----
    Assumes 4 clusters
    """

    # whiten the theta phases to normalize the data
    # theta_phases = whiten(theta_phases)
    # Perform k-means clustering on theta phases
    rng = np.random.default_rng(42)
    km = kmeans2(theta_phases, 4, rng=rng, minit="points")

    cluster_means = km[0]
    cluster_labels = km[1]

    targets = {"Trough": 0, "Rising": 90, "Peak": 180, "Falling": 270}

    target_rads = {k: np.deg2rad(v) for k, v in targets.items()}

    angs = np.linspace(0, 2 * np.pi, 36)
    cluster_means = {}
    for label in range(4):
        angles = theta_phases[cluster_labels == label]
        cluster_means[label] = circ_mean(angs, angles)

    assignments = {}
    used_labels = set()
    for target, target_angle in target_rads.items():
        # Find unused cluster label with closest mean angle
        best_label = min(
            (l for l in cluster_means if l not in used_labels),
            key=lambda l: np.abs(
                np.angle(np.exp(1j * (cluster_means[l] - target_angle)))
            ),
        )
        assignments[target] = best_label
        used_labels.add(best_label)

    mapping = dict((v, k) for k, v in assignments.items())
    lookup = [mapping[i] for i in range(4)]
    labels = np.take(lookup, cluster_labels)

    return km, labels, mapping


def plot_theta_phase_dist(km_result: tuple, assignments: dict):
    """
    Plot the distribution of theta phases for the two cell types
    in df. The dataframe should have a column "theta_phase" and
    "cell_type".

    Parameters
    ----------
    km_result - resuilt of kmeans2 (see get_theta_labels above)
    assignments - unorderd dict mapping integer labels in km_result
                to the phase of LFP theta at which the cluster has its
                maximum
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(nrows=len(km_result[0]) + 1, ncols=1, figsize=(8, 6))
    plt.suptitle(
        "k-means clustering of theta phase of firing\ntheta phase histograms",
        fontweight="bold",
        fontsize=14,
    )

    degs = np.linspace(0, 360 * 2, km_result[0][0].shape[0] * 2)
    cols = sns.color_palette("tab10")[0 : len(km_result[0])]

    # sort the dictionary
    phase_ids = OrderedDict(sorted(assignments.items()))

    # TODO: add the number of items of each label
    for k, v in phase_ids.items():
        axs[k].set_xlim(0, 720)
        h = km_result[0][k]
        h = h / np.sum(h)
        ax_label = f"{v}\nn={np.count_nonzero(km_result[1] == k)}"
        axs[k].plot(degs, np.tile(h, 2), label=ax_label, color=cols.pop(0))
        axs[k].set_yticks([])
        axs[k].set_xticks([])
        axs[k].set_yticklabels([])
        axs[k].set_xticklabels([])
        axs[k].spines["left"].set_visible(False)
        axs[k].spines["right"].set_visible(False)
        axs[k].spines["top"].set_visible(False)
        axs[k].legend()

    ax = axs[-1]
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(0, 720)
    ax.plot(degs, -np.cos(np.deg2rad(degs)), color="k")
    plt.xlabel("Theta phase (degrees)")

    return fig
