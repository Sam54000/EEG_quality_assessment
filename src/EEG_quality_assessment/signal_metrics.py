#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-03-06
# email: samuel DOT louviot AT nki DOT rfmh DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================================
"""MODULE DESCRIPTION HERE."""
# standard library imports

# third-party imports (and comments indicating how to install them)
# python -m conda install -c conda-forge mne or python -m pip install mne
import mne

# python -m conda install -c conda-forge numpy or python -m pip install numpy
import numpy as np

# python -m conda install -c conda-forge scipy or python -m pip install scipy
import scipy


def average_rms(signal: np.ndarray) -> float:
    """Calculate the average root mean square of the signal.

    The RMS of a signal is regarded as the magnitude of it.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array

    Returns:
        float: the average root mean square of the signal
    """
    return np.sqrt(np.mean(signal**2))


def max_gradient(signal: np.ndarray) -> float:
    """Calculate the maximum gradient of the signal.

    The maximum gradient is the maximum absolute value
    between 2 consecutive values of the signal for 2 consecutive
    time samples. It is usefull to detect high amplitude,
    high frequency artifacts.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array

    Returns:
        float: the maximum gradient of the signal
    """
    return np.max(np.abs(np.diff(signal)))


def zero_crossing_rate(signal: np.ndarray) -> float:
    """Calculate the zero crossing rate of the signal.

    It is the rate at which the signal cross the 0 line.
    High frequency signal will have a high rate,
    and low frequency/drifting signal will have a low rate.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array

    Returns:
        float: the zero crossing rate of the signal
    """
    return np.mean(np.diff(np.sign(signal) != 0))


def hjorth_mobility(signal: np.ndarray) -> float:
    """Calculate the mobility from the Hjorth parameters.

    The mobility is a measure of the signal's frequency content.
    It is the ratio of the standard deviation of the derivative of the signal
    to the standard deviation of the signal.

    Args:
        signal (np.ndarrary): the signal to be analyzed
                              has to be a 1D array.

    Returns:
        float: the mobility score of the signal
    """
    derived_signal_variance = np.var(np.diff(signal))
    signal_variance = np.var(signal)
    return np.sqrt(derived_signal_variance / signal_variance)


def hjorth_complexity(signal: np.ndarray) -> float:
    """Calculate the complexity from the Hjorth parameters.

    The complexity, as it is indicated by the name, is a measure of the
    complexity of the signal. It is the ratio of the mobility of the derivative
    of the signal to the mobility of the signal.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.

    Returns:
        float: the complexity score of the signal
    """
    derived_signal_mobility = hjorth_mobility(np.diff(signal))
    signal_mobility = hjorth_mobility(signal)
    return derived_signal_mobility / signal_mobility


def kurtosis(signal: np.ndarray) -> float:
    """Calculate the kurtosis of the signal.

    The kurtosis is a measure of the "tailedness" of the signal.
    In other words it is a measure of how heavy the tails of the signal value
    distribution are. A high kurtosis means that the tails are heavy thus the
    signal has a lof of outliers. A low kurtosis means that the tails are light
    and the signal has less outliers.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.

    Returns:
        float: _description_
    """
    return scipy.stats.kurtosis(signal)


def skewness(signal: np.ndarray) -> float:
    """Calculate the skewness of the signal.

    The skewness is a measure of the asymmetry of the signal.
    A positive skewness means that the signal is skewed to the right,
    a negative skewness means that the signal is skewed to the left.
    It is a complementary measure to the kurtosis.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.

    Returns:
        float: the skewness of the signal
    """
    return scipy.stats.skew(signal)


def variance(signal: np.ndarray) -> float:
    """Calculate the variance of the signal.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.

    Returns:
        float: the variance of the signal
    """
    return np.var(signal)


def signal_range(signal: np.ndarray) -> float:
    """Calculate the range of the signal.

    Range of the signal is the difference between the maximum and the minimum
    values.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.

    Returns:
        float: the range of the signal
    """
    return np.max(signal) - np.min(signal)


def signal_IQR(signal: np.ndarray) -> float:
    """Calculate the interquartile range of the signal.

    The interquartile range is the range of the middle 50% of the signal.
    It is less sensitive to outliers than the range.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.

    Returns:
        float: the interquartile range of the signal
    """
    return scipy.stats.iqr(signal)


# No time window needed. I can deal with mne object now
# TO FINISH
def epochs_snr(epochs: mne.Epochs) -> mne.EvokedArray:
    """Calculate the signal to noise ratio of an Evoked Related Potential.

    What is considered here as the signal is the ERP (the average signal
    across epochs). The noise is the standard deviation across epochs.

    Args:
        epochs (mne.Epochs): An mne.Epochs object containing the epochs

    Returns:
        mne.EvokedArray: An mne.EvokedArray object containing the signal to noise
                         ratio of the ERP
    """
    erp_signal = epochs.copy().average().get_data()
    erp_noise = epochs.copy().get_data().std(axis=0)
    snr = np.divide(erp_signal**2, erp_noise**2)
    snr_decibel = 10 * np.log10(snr)
    snr_decibel_mne_object = mne.EvokedArray(snr_decibel, epochs.info)
    return snr_decibel_mne_object


# TODO
# - Make a class object to store the steps of the process
# in order to keep a history of what has been done.
# - The argument "frequency_of_interest" doesn't respect the dry principle
# I should refactor this in order to have to declare the frequency only once.
# - Create a subclass for the different spectrum types (amplitude, zscore, snr, phase)
