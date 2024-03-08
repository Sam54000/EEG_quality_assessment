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
# This program is free software: you can redistribute it and/or modify
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
import copy
from typing import Any, Callable

# third-party imports (and comments indicating how to install them)
# python -m conda install -c conda-forge mne or python -m pip install mne
import mne

# python -m conda install -c conda-forge numpy or python -m pip install numpy
import numpy as np


class FourierSpectrum:
    """A class to store and manipulate the Fourier Spectrum of a signal."""

    def __init__(self, raw: mne.io.Raw) -> None:
        """Initialize the FourierSpectrum object.

        Args:
            raw (mne.io.Raw): the raw signal to be analyzed
                              has to be an mne.io.Raw object
        """
        self.signal = raw.get_data()
        self.sampling_rate = raw.info["sfreq"]
        self._adjust_signal_length()
        self.fft = np.fft.fft(self.signal)
        self.spectrum = np.abs(self.fft)
        self.frequencies = np.fft.fftfreq(len(self.signal), 1 / self.sampling_rate)
        self.frequencies = self.frequencies[: len(self.frequencies) // 2]
        self.spectrum = self.spectrum[: len(self.spectrum) // 2]
        self.spectrum_type = "raw amplitude"
        self.frequency_resolution = self.frequencies[1] - self.frequencies[0]
        del self.signal  # to save memory because dealing with high sampling rate

    def _adjust_signal_length(self) -> 'FourierSpectrum':
        """Adjust the signal length to be a power of 2.

        This is done in order to have a faster computation of the fft.

        Args:
        signal (numpy.ndarray): A 2D numpy array (channels, time).

        Returns:
        Self: the FourierSpectrum object
        """
        current_length = self.signal.shape[1]

        if current_length & (current_length - 1) != 0:
            new_length = 2 ** int(np.log2(current_length))
            np.resize(self.signal, self.signal[:, :new_length].shape)

        return self

    def set_frequency_of_interest(self, frequency_of_interest: float) -> 'FourierSpectrum':
        """Set the frequency of interest.

        This is following the methodology of Rossion et al. 2014 and Jonas et al. 2016
        in order to get the amplitude of the surrounding bins.

        Args:
            frequency_of_interest (float): The frequency of interest.

        Returns:
            typing.Self: The FourierSpectrum object
        """
        self.frequency_of_interest = frequency_of_interest
        return self

    def check_if_frequency_of_interest_exists(func: Any) -> Callable[..., Any]:
        """Check if the frequency of interest has been set before using the method.

        Args:
            func (Any): the method to be checked
        """

        def wrapper(self: 'FourierSpectrum',
                    *args: tuple,
                    **kwargs: dict) -> 'FourierSpectrum':

            if hasattr(self, "frequency_of_interest"):
                return func(self, *args, **kwargs)

            else:
                raise ValueError(
                    """
                The frequency of interest has to be set before using this method.
                Use the set_frequency_of_interest() method to set it.
                """
                )

        return wrapper

    def _get_frequency_index(self, frequency_of_interest: float) -> int:
        """Get the index of the frequency of interest in the spectrum.

        Args:
            frequency_of_interest (float): in Hertz

        Returns:
            int: The index (position) in the spectrum
        """
        return np.argmin(np.abs(self.frequencies - frequency_of_interest))

    def _get_amplitude_surounding_bins(
        self,
        desired_frequency_step: float = 0.0135,
        nb_steps: int = 25,
    ) -> np.ndarray:
        """Get the amplitude of the surrounding bins of the frequency of interest.

        This is inspired from the Rossion et al. 2014
        and Jonas et al. 2016 papers who are pioneer in face recognition
        Frequency Tagging of Steady State Visual Evoked Potential -> SSVEP
        (more precisely Fast Periodic Visual Stimulation -> FPVS).
        They perform calculation based on the 50 surrounding bins of the frequency
        of interest (25 bins before and the 25 bins after the frequency of interest).
        They leave out one bin before and one bin after the frequency of interest.
        Their bins are for a fourier spectrum from a signal sampled at 512 Hz therefore
        1 bin is 0.0135 Hz. In order to reproduce the same calculation as in the articles
        it is thought in terms of frequency steps instead of bins due to the possiblity
        of having different sampling rates and frequency resolutions.

        Args:
            frequency_of_interest (float): The frequency around
            desired_frequency_step (float, optional): _description_. Defaults to 0.0135
                                                      to reproduce the articles
                                                      methodology.
            nb_steps (int, optional): _description_. Defaults to 25.

        Returns:
            np.ndarray: _description_
        """
        frequency_index = self._get_frequency_index(self.frequency_of_interest)

        nb_bins = int(desired_frequency_step / self.frequency_resolution)
        amplitude_surrounding_left_bins = self.spectrum[
            frequency_index - nb_steps * nb_bins : frequency_index - nb_bins
        ]

        amplitude_surrounding_right_bins = self.spectrum[
            frequency_index + nb_bins : frequency_index + nb_steps * nb_bins
        ]

        return np.concatenate(
            (amplitude_surrounding_left_bins, amplitude_surrounding_right_bins)
        )

    @check_if_frequency_of_interest_exists
    def _get_baseline(self) -> float:
        """Get the baseline of the spectrum around the frequency of interest.

        This baseline calculation is inspired from the Rossion et al. 2014
        and Jonas et al. 2016 papers who are pioneer in face recognition
        Frequency Tagging of Steady State Visual Evoked Potential -> SSVEP
        (more precisely Fast Periodic Visual Stimulation -> FPVS).
        The baseline is considered as the mean amplitude of then n surrounding
        bins of the frequency of interest.
        In the articles they consider the 50 surrounding bins (25 bins before
        and the 25 bins after the frequency of interest).They leave out one
        bin before and one bin after the frequency of interest. Then, they compute
        the mean of the 48 remaining bins (24 on the left and 24 on the right
        of the frequency of interest). The sampling rate used in their articles
        is 512 Hz so 1 bin is 0.0135 Hz.

        Returns:
            float : The baseline value

        .. _Rossion et al. 2014: https://pubmed.ncbi.nlm.nih.gov/24728131/
        .. _Jonas et al. 2016: https://pubmed.ncbi.nlm.nih.gov/27354526/

        """
        amplitude_surrounding_bins = self._get_amplitude_surounding_bins(
            self.frequency_of_interest
        )
        return np.mean(amplitude_surrounding_bins)

    @check_if_frequency_of_interest_exists
    def correct_baseline(self) -> 'FourierSpectrum':
        """Remove the baseline of the spectrum.

        It removes the baseline on the entire spectrum. What is considered
        as the baseline is the mean amplitude of the surrounding bins of the
        frequency of interest (see _get_baseline() method for more details).
        WARNING: This method is not reversible and will modify the spectrum.
        Consider using the copy() method before using this method if you want
        to keep the original spectrum.

        Args:
            frequency_of_interest (float, optional): The frequency around which
                                                     the baseline, snr and zscore
                                                     will be calculated. Defaults to 12.

        Returns:
            self : The modified FourierSpectrum object.
        """
        baseline = self._get_baseline(self.frequency_of_interest)
        self.spectrum_type = "baseline corrected amplitude"
        self.spectrum = self.spectrum - baseline
        return self

    @check_if_frequency_of_interest_exists
    def calculate_zscore(self) -> 'FourierSpectrum':
        """Calculate the zscore of the spectrum.

        It follows the methodology of Rossion et al. 2014 and Jonas et al. 2016.
        The zscore is calculated as the amplitude of the frequency of interest divided
        by the standard deviation of the amplitude of the surrounding bins.

        Returns:
            Self: The modified FourierSpectrum object.
        """
        if self.spectrum_type != "baseline corrected amplitude":
            self.correct_baseline(self.frequency_of_interest)

        amplitude_surrounding_bins = self._get_amplitude_surounding_bins(
            self.frequency_of_interest
        )
        std = np.std(amplitude_surrounding_bins)
        self.spectrum_type = "zscore"
        self.spectrum = self.spectrum / std

        return self

    @check_if_frequency_of_interest_exists
    def calculate_snr(self) -> 'FourierSpectrum':
        """Calculate the signal to noise ratio of the spectrum.

        Following the methodology of Rossion et al. 2014 and Jonas et al. 2016.
        The signal to noise ratio is calculated as the baseline corrected amplitude
        of the frequency of interest divided by the baseline amplitude.

        Returns:
            Self: The modified FourierSpectrum object.
        """
        if self.spectrum_type == "amplitude":
            baseline = self._get_baseline(self.frequency_of_interest)
            self.spectrum_type = "snr"
            self.spectrum = self.spectrum / baseline
        else:
            raise TypeError(
                f"""
            The spectrum has to be an amplitude spectrum to calculate the snr.
            Got a {self.spectrum_type} spectrum instead. Please re instantiate
            the FourierSpectrum object with the raw signal.
            """
            )

        return self

    def calculate_phase(self) -> 'FourierSpectrum':
        """Calculate the phase of the spectrum.

        Returns:
            Self: The modified FourierSpectrum object.
        """
        self.spectrum_type = "phase"
        self.phase = np.angle(self.fft)
        return self

    def copy(self) -> 'FourierSpectrum':
        """Copy the object following the same philosophy as mne objects.

        Returns:
            self: A copy of the instance
        """
        return copy.copy(self)


# TODO
# - Think about making a subclass for the different
#   spectrum types (amplitude, zscore, snr, phase)
