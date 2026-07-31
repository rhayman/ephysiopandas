import numpy as np
import matplotlib.pylab as plt
from ephysiopy.io.recording import TrialInterface as Trial
from ephysiopandas.data_io import get_LFPOscillation


def theta_spectrum_max(trial: Trial, **kws) -> float:
    """
    Get the maximum power frequency in the theta band from the
    LFP power spectrum.

    Parameters
    ----------
    trial (TrialInterface) - the trial object

    kws - keyword arguments
        theta_band (list) - the frequency range for theta band
                            default: [6, 12]
        low_speed (float) - the low speed threshold for power spectrum
                            calculation. default: 2 cm/s
    """

    low_speed = kws.get("low_speed", 2)
    high_speed = kws.get("high_speed", 30)

    LFP = get_LFPOscillation(trial)
    LFP.thetaRange = kws.get("theta_band", [6, 12])

    # get the max power frequency in the theta band
    power_spectrum = LFP.power_spectrum(
        low_speed=low_speed, high_speed=high_speed, plot=False
    )
    return power_spectrum["maxFreq"]


def n_oscillatory_epochs(trial: Trial, **kws) -> int:
    """
    Get the number of oscillatory epochs in the specified frequency band.

    Parameters
    ----------
    trial (TrialInterface) - the trial object

    kws - keyword arguments
        freq_band (list) - the frequency range for oscillatory epoch detection
                            default: [20, 90] (gamma band)

    Returns
    -------
    int - number of oscillatory epochs in the specified frequency band

    """

    freq_band = kws.get("freq_band", [20, 90])  # default gamma band
    LFP = get_LFPOscillation(trial)
    n_gamma_power_events = len(LFP.get_oscillatory_epochs(FREQ_BAND=freq_band))

    return n_gamma_power_events


def theta_gamma_modulation_index(trial: Trial, **kws) -> float:
    """
    Calculate the modulation index for theta-gamma coupling.

    Parameters
    ----------
    trial (TrialInterface) - the trial object
    kws - keyword arguments
        theta_band (list) - the frequency range for theta band
                            default: [6, 12]
        gamma_band (list) - the frequency range for gamma band
                            default: [30, 90]
    """

    theta_band = kws.get("theta_band", [6, 12])
    gamma_band = kws.get("gamma_band", [30, 90])

    LFP = get_LFPOscillation(trial)
    LFP.thetaRange = theta_band

    return LFP.modulationindex(gammaband=gamma_band)


def comodulogram_PAC(trial: Trial, **kws) -> dict:
    """
    Calculate the phase-amplitude coupling (PAC) value from the comodulogram.

    Returns the maximas in the slow and fast gamma bands at the theta frequency range.

    Parameters
    ----------
    trial (TrialInterface) - the trial object
    kws - keyword arguments
        theta_band (list) - the frequency range for theta band
                            default: [6, 12]
        slow_gamma (list) - the frequency range for the slow gamma band
                            default: [20, 40]
        fast_gamma (list) - the frequency range for the fast gamma band
                            default: [40, 90]

    Returns
    -------
    dict - the max PAC values from the comodulogram for the slow and
            fast gamma bands at the theta frequency range
            keys: 'slow_gamma' and 'fast_gamma'
    """

    theta_band = kws.get("theta_band", [6, 12])
    slow_gamma = kws.get("slow_gamma", [20, 40])
    fast_gamma = kws.get("fast_gamma", [40, 90])

    LFP = get_LFPOscillation(trial)
    LFP.thetaRange = theta_band

    C = LFP.get_comodulogram(low_freq_band=theta_band, **kws)

    y, x = np.meshgrid(C.high_fq_range, C.low_fq_range)

    slow_mask = np.logical_and(
        np.logical_and(x >= theta_band[0], x <= theta_band[1]),
        np.logical_and(y >= slow_gamma[0], y <= slow_gamma[1]),
    )
    fast_mask = np.logical_and(
        np.logical_and(x >= theta_band[0], x <= theta_band[1]),
        np.logical_and(y >= fast_gamma[0], y <= fast_gamma[1]),
    )

    return {
        "slow_gamma": np.nanmax(C.comod_[slow_mask]),
        "fast_gamma": np.nanmax(C.comod_[fast_mask]),
    }


def PAC_delay_estimate(trial: Trial, **kws) -> plt.Figure:
    """
    Estimate the delay between theta phase and gamma amplitude using the
    phase-amplitude coupling (PAC) analysis.

    Parameters
    ----------
    trial (TrialInterface) - the trial object
    kws - keyword arguments
        theta_band (list) - the frequency range for theta band
                            default: [6, 12]
        gamma_band (list) - the frequency range for gamma band
                            default: [30, 90]

    Returns
    -------
    float - the estimated delay between theta phase and gamma amplitude

    """

    theta_band = kws.get("theta_band", [6, 12])
    gamma_band = kws.get("gamma_band", [30, 90])

    LFP = get_LFPOscillation(trial)
    LFP.thetaRange = theta_band

    return LFP.PAC_delay_estimate(gamma_band=gamma_band)


def max_theta_freq(trial: Trial, **kws) -> float:
    """
    Get the maximum frequency in the theta band from the power spectrum.

    Parameters
    ----------
    trial (TrialInterface) - the trial object
    kws - keyword arguments
        theta_band (list) - the frequency range for theta band
                            default: [6, 12]

    Returns
    -------
    float - the maximum frequency in the theta band from the power spectrum
    """

    theta_band = kws.get("theta_band", [6, 12])
    exclude_speeds = kws.get("exclude_speeds", [0, 2])
    high_speed_threshold = kws.get("high_speed_threshold", 30)

    LFP = get_LFPOscillation(trial)
    LFP.thetaRange = theta_band

    power_spectrum = LFP.power_spectrum(
        low_speed=exclude_speeds[1], high_speed=high_speed_threshold, plot=False
    )

    return power_spectrum["maxFreq"]


def theta_running(trial: Trial, **kws) -> float:
    """
    Calculate the average theta power during running epochs.

    Parameters
    ----------
    trial (TrialInterface) - the trial object
    kws - keyword arguments
        theta_band (list) - the frequency range for theta band
                            default: [6, 12]
        running_speed_threshold (float) - the speed threshold for defining running epochs
                                        default: 5 cm/s

    Returns
    -------
    dict - the results of the linear regression of speed vs theta frequency

    See Also
    --------
    scipy.stats.linregress
    """

    theta_band = kws.get("theta_band", [6, 12])
    low_speed_threshold = kws.get("low_speed_threshold", 0.5)
    high_speed_threshold = kws.get("high_speed_threshold", 30)

    LFP = get_LFPOscillation(trial)
    LFP.thetaRange = theta_band

    res, _, _ = LFP.theta_running(
        trial.PosCalcs,
        trial.EEGCalcs,
        high_speed=high_speed_threshold,
        low_speed=low_speed_threshold,
        plot=False,
    )

    return res._asdict()
