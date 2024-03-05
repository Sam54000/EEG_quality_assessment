import mne
import numpy as np
import scipy


def average_rms(signal: np.ndarray) -> float:
    """Calculate the average root mean square of the signal.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array

    Returns:
        float: the average root mean square of the signal
    """
    return np.sqrt(np.mean(signal**2))


def max_gradient(signal: np.ndarray) -> float:
    """Calculate the maximum gradient of the signal.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array

    Returns:
        float: the maximum gradient of the signal
    """
    return np.max(np.abs(np.diff(signal)))


def zero_crossing_rate(signal: np.ndarray) -> float:
    """Calculate the zero crossing rate of the signal.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array

    Returns:
        float: the zero crossing rate of the signal
    """
    return np.mean(np.diff(np.sign(signal) != 0))


def hjorth_mobility(signal):
    derived_signal_variance = np.var(np.diff(signal))
    signal_variance = np.var(signal)
    return np.sqrt(derived_signal_variance / signal_variance)


def hjorth_complexity(signal):
    derived_signal_mobility = hjorth_mobility(np.diff(signal))
    signal_mobility = hjorth_mobility(signal)
    return derived_signal_mobility / signal_mobility


def kurtosis(signal):
    return scipy.stats.kurtosis(signal)


def skewness(signal):
    return scipy.stats.skew(signal)


def signal_variance(signal):
    return np.var(signal)


def signal_range(signal):
    return np.max(signal) - np.min(signal)


def signal_IQR(signal):
    return scipy.stats.iqr(signal)


# No time window needed. I can deal with mne object now
# TO FINISH
# def epochs_snr(epochs, noise):
#    return 10*np.log10(np.mean(signal**2)/np.mean(noise**2))
#
# def spectrum_snr(frequency_spectrum, studied_frequency = 12):
#    pass
