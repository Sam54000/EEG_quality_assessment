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
import typing

# third-party imports (and comments indicating how to install them)
# python -m conda install -c conda-forge mne or python -m pip install mne
import mne

# python -m conda install -c conda-forge numpy or python -m pip install numpy
import numpy as np

# python -m conda install -c conda-forge scipy or python -m pip install scipy
import scipy


def average_rms(signal: np.ndarray, axis = 1) -> float:
    """Calculate the average root mean square of the signal.

    The RMS of a signal is regarded as the magnitude of it.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array
        mean_kwargs (dict): the keyword arguments to be passed to the np.mean

    Returns:
        float: the average root mean square of the signal
    """
    return np.sqrt(np.mean(signal**2, axis=axis))


def max_gradient(signal: np.ndarray,axis:int = 1) -> float:
    """Calculate the maximum gradient of the signal.

    The maximum gradient is the maximum absolute value
    between 2 consecutive values of the signal for 2 consecutive
    time samples. It is usefull to detect high amplitude,
    high frequency artifacts.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array
        axis (int): the axis along which the gradient is calculated

    Returns:
        float: the maximum gradient of the signal
    """
    return np.max(np.abs(np.diff(signal, axis = axis)), axis=axis)


def zero_crossing_rate(signal: np.ndarray, axis:int = 1) -> float:
    """Calculate the zero crossing rate of the signal.

    It is the rate at which the signal cross the 0 line.
    High frequency signal will have a high rate,
    and low frequency/drifting signal will have a low rate.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array
        axis (int): the axis along which the zero crossing rate is calculated

    Returns:
        float: the zero crossing rate of the signal
    """
    return np.mean(np.diff(np.sign(signal) != 0, axis = axis), axis = axis)


def hjorth_mobility(signal: np.ndarray,axis:int = 1) -> float:
    """Calculate the mobility from the Hjorth parameters.

    The mobility is a measure of the signal's frequency content.
    It is the ratio of the standard deviation of the derivative of the signal
    to the standard deviation of the signal.

    Args:
        signal (np.ndarrary): the signal to be analyzed
                              has to be a 1D array.
        axis (int): the axis along which the mobility is calculated

    Returns:
        float: the mobility score of the signal
    """
    derived_signal_variance = np.var(np.diff(signal, axis = axis), axis = axis)
    signal_variance = np.var(signal, axis = axis)
    return np.sqrt(derived_signal_variance / signal_variance)


def hjorth_complexity(signal: np.ndarray,axis:int = 1) -> np.ndarray:
    """Calculate the complexity from the Hjorth parameters.

    The complexity, as it is indicated by the name, is a measure of the
    complexity of the signal. It is the ratio of the mobility of the derivative
    of the signal to the mobility of the signal.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.
        axis (int): the axis along which the complexity is calculated

    Returns:
        float: the complexity score of the signal
    """
    derived_signal_mobility = hjorth_mobility(np.diff(signal,axis = axis),
                                                axis = axis)
    signal_mobility = hjorth_mobility(signal,axis = axis)
    return derived_signal_mobility / signal_mobility


def signal_range(signal:np.ndarray, axis:int = 1) -> float:
    """Calculate the range of the signal.

    Range of the signal is the difference between the maximum and the minimum
    values.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.
        axis (int): the axis along which the range is calculated

    Returns:
        float: the range of the signal
    """
    return np.subtract(np.max(signal, axis = axis),
                       np.min(signal, axis = axis))


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
# - Think about epoching the gradient peak.
