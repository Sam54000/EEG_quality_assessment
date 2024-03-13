#!/usr/bin/env -S  python  #
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
"""This module handle frequency spectrum obtained from an FFT.

The different calculations perfomed on the spectrum
follow a methodology used in Fast Periodic Visual Stimulation (FPVS) to
quantify with a high Signal Noise Ratio some cognitive processes at a specific
frequency (e.g. the frequency of an Oddball). It is inspired from the Rossion
et al. 2014 and Jonas et al. 2016.

Note:
    The articles mentionned perform a Steady State Evoked Related Potential
    (SSVEP) paradigm which is called Fast Periodic Visual Stimulation
    (Rossion et al. 2014). The paradigm involves a visual stimulation (images)
    at a base frequency (6 Hz). Every 5 images, an oddball is pressented
    (A face among non face images, or a known face among uknown faces etc.). The
    odball frequency is 6/5 = 1.2 Hz. In order to quantify the
    electrophysiological response of the cognitive process exhibited by the
    oddball, the authors perform a frequency tagging analysis of the signal.
    They calculate the fast fourier transform (FFT) of the signal and then
    perform different operations and claculations on the spectrum.
    The different operations are the correction of the baseline, the zscore
    (to quantify the significance of the amplitude of the frequency of interest
    from the "noise"), and the signal to noise ratio. To do so, they consider a
    specific zone of 50 surrounding bins around the frequency of interest
    (25 bins before and 25 bins after the frequency of interest).
    They leave out one bin before and one bin after the frequency
    of interest that lead to 24 bins each side of the frequency of interest.
    This zone is consider as the "surrounding noise" of the brain activity and
    the amplitude of the frequency of interest is considered as the "signal".
    Therefore, the zscore is calculated as the amplitude of the signal divided
    by the standard deviation of the surrounding noise. The signal to noise
    ratio is calculated as the amplitude of the signal divided by the mean
    amplitude of the surrounding noise.
    Their methdology was implemented for a signal sampled at 512 Hz. Therefore
    1 bin = 0.0135 Hz. In this module the approach used is in frequency step
    rather than frequency bin, in order to have the same frequency
    values as in the articles still offering flexibility in term of input
    sampling frequency and frequency resolution.

    This whole approach is interesting to quantify also noise that are at a
    known frequency (e.g. 60 Hz noise from the electrical grid, gradient
    artifacts for EEG-fMRI).

    .. _Rossion et al. 2014: https://pubmed.ncbi.nlm.nih.gov/24728131/
    .. _Jonas et al. 2016: https://pubmed.ncbi.nlm.nih.gov/27354526/

"""
# standard library imports
import copy
from typing import Any, Callable

# third-party imports (and comments indicating how to install them)
# python -m conda install -c conda-forge mne or python -m pip install mne
import mne

# python -m conda install -c conda-forge numpy or python -m pip install numpy
import numpy as np


class Spectrum:
    """A class to store and manipulate the Fourier Spectrum of a signal."""

    def __init__(self) -> None:
        """Initialize."""
        self.process_history: list = list()

    def calculate_fft(self, raw: mne.io.Raw) -> 'Spectrum':
        """Initialize the Spectrum object.

        Args:
            raw (mne.io.Raw): The raw signal to be analyzed
        """
        self.sampling_rate = raw.info["sfreq"]
        self.signal = raw.get_data()
        self._adjust_signal_length()
        self.spectrum = np.fft.fft(self.signal)
        spectrum_length = np.shape(self.signal)[1]
        self.frequencies = np.fft.fftfreq(spectrum_length,
                                          1 / self.sampling_rate)
        self.frequencies = self.frequencies[: len(self.frequencies) // 2]
        self.spectrum = self.spectrum[:,:spectrum_length// 2]
        self.frequency_resolution = self.frequencies[1] - self.frequencies[0]
        return self

    def _adjust_signal_length(self) -> 'Spectrum':
        """Adjust the signal length to be a power of 2.

        This is done in order to have a faster computation of the fft.

        Args:
        signal (numpy.ndarray): A 2D numpy array (channels, time).

        Returns:
        Self: the Spectrum object
        """
        current_length = self.signal.shape[1]

        if current_length & (current_length - 1) != 0:
            new_length = 2 ** int(np.log2(current_length))
            np.resize(self.signal, self.signal[:, :new_length].shape)

        self.process_history.append("Signal length adjusted to be a power of 2")

        return self

    def _set_frequency_of_interest(self,
                                  frequency_of_interest: float = 12
                                  ) -> 'Spectrum':
        """Set the frequency of interest.

        This is following the methodology of Rossion et al. 2014 and Jonas et al. 2016
        in order to get the amplitude of the surrounding bins.

        Args:
            frequency_of_interest (float): The frequency of interest.

        Returns:
            typing.Self: The Spectrum object
        """
        self.frequency_of_interest = frequency_of_interest
        self.process_history.append(
            f"Frequency of interest set to {frequency_of_interest}"
            )
        return self

    def _frequency_of_interest_exists(self) -> bool:
        """Check if the frequency of interest has been set.

        Returns:
            bool: True if the frequency of interest has been set, False otherwise.
        """
        return hasattr(self, "frequency_of_interest")

    def _get_frequency_index(self, frequency: float) -> int:
        """Get the index of the frequency of interest in the spectrum.

        Args:
            frequency (float): in Hertz

        Returns:
            int: The index (position) in the spectrum
        """
        return np.argmin(np.abs(self.frequencies - frequency))

    def _get_amplitude_surounding_bins(
        self,
        desired_frequency_step: float = 0.0135,
        nb_steps: int = 25,
    ) -> np.ndarray:
        """Get the amplitude of the surrounding bins of the frequency of interest.

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
        amplitude_surrounding_left_bins = self.spectrum[:,
            frequency_index - nb_steps * nb_bins : frequency_index - nb_bins
        ]

        amplitude_surrounding_right_bins = self.spectrum[:,
            frequency_index + nb_bins : frequency_index + nb_steps * nb_bins
        ]

        self.process_history.append(
            f"""Amplitude of the {(nb_steps-1)*2} surrounding bins of the
            frequency of interest calculated with a frequency step of
            {desired_frequency_step} Hz"""
        )

        return np.concatenate(
            (amplitude_surrounding_left_bins, amplitude_surrounding_right_bins),
            axis=1
        )

    def _get_baseline(self) -> float:
        """Get the baseline of the spectrum around the frequency of interest.

        The baseline is the mean amplitude of the surrounding bins of the
        frequency of interest.

        Returns:
            float : The baseline value

        """
        if not self._frequency_of_interest_exists():
            raise ValueError("The frequency of interest has to be set first.")
        amplitude_surrounding_bins = self._get_amplitude_surounding_bins()
        return np.mean(amplitude_surrounding_bins)

    def correct_baseline(self) -> 'Spectrum':
        """Remove the baseline of the spectrum.

        It removes the baseline on the entire spectrum. What is considered
        as the baseline is the mean amplitude of the surrounding bins of the
        frequency of interest (see _get_baseline() method for more details).
        WARNING: This method is not reversible and will modify the spectrum.
        Consider using the copy() method before using this method if you want
        to keep the original spectrum.

        Args:
            frequency_of_interest (float, optional): The frequency around which
                                                     the baseline, snr and
                                                     zscore will be calculated.
                                                     Defaults to 12.

        Returns:
            self : The modified Spectrum object.
        """
        baseline = self._get_baseline()
        self.process_history.append("Baseline corrected")
        self.spectrum = self.spectrum - baseline
        return self

    def _baseline_corrected(self) -> bool:
        """Check if the baseline has been corrected.

        Returns:
            bool: True if the baseline has been corrected, False otherwise.
        """
        return 'basline corrected' in self.process_history

    def calculate_amplitude(self) -> 'Spectrum':
        """Create an AmplitudeSpectrum object from a Spectrum object.

        Args:
            spectrum (Spectrum): The Spectrum object
                                        to be converted.

        Returns:
            AmplitudeSpectrum: The AmplitudeSpectrum object.
        """
        self.spectrum = np.abs(self.spectrum)
        return self

    def calculate_zscore(self) -> 'Spectrum':
        """Calculate the zscore of the spectrum.

        The zscore is calculated as the amplitude of the frequency of interest
        divided by the standard deviation of the amplitude of the surrounding
        bins.

        Args:
            spectrum (Spectrum): The Spectrum object to be
                                        converted.
            frequency_of_interest (int, optional): The frequency around
                                                   which the standard
                                                   deviation is calculated.
                                                   Defaults to 12.

        Raises:
            TypeError: The input has to be an AmplitudeSpectrum to calculate
                       the zscore.

        Returns:
            ZscoreSpectrum: The ZscoreSpectrum object.
        """
        if not self._frequency_of_interest_exists():
            raise ValueError("The frequency of interest has to be set first.")

        if not self._baseline_corrected():
            self.correct_baseline()

        amplitude_surrounding_bins = self._get_amplitude_surounding_bins()

        surrounding_bin_std = np.std(amplitude_surrounding_bins)
        self.spectrum = np.divide(
            self.spectrum,
            surrounding_bin_std
        )

        return self

    def calculate_snr(self) -> 'Spectrum ':
        """Calculate the signal to noise ratio of the spectrum.

        The signal to noise ratio is calculated as the baseline corrected
        amplitude of the frequency of interest divided by the baseline
        amplitude.

        Args:
            spectrum (Spectrum): The Spectrum object to be
                                        converted.
            frequency_of_interest (int, optional): The frequency around
                                                   which the baseline and
                                                   snr will be calculated.
                                                   Defaults to 12.

        Raises:
            TypeError: The input has to be an AmplitudeSpectrum to calculate
                       the snr.

        Returns:
            SnrSpectrum: The SnrSpectrum object.
        """
        if not self._frequency_of_interest_exists():
            raise ValueError("The frequency of interest has to be set first.")

        baseline = self._get_baseline()
        self.spectrum = np.divide(
            self.spectrum,
            baseline
        )
        return self

    def calculate_phase(self) -> 'Spectrum':
        """Calculate the phase of the spectrum.

        Returns:
            Self: The modified Spectrum object.
        """
        self.phase = np.angle(self.spectrum)
        return self

    def copy(self) -> 'Spectrum':
        """Copy the object following the same philosophy as mne objects.

        Returns:
            self: A copy of the instance
        """
        return copy.copy(self)

    def get_magnitude_at_frequency(self,
                                   frequency: float,
                                   margin_frequency: float = 1) -> float:
        """Get the amplitude at a specific frequency.

        Args:
            frequency (float): The frequency of interest.
            margin_frequency (float, optional): The margin around the frequency
                                               of interest to calculate the
                                               amplitude. Defaults to None.

        Returns:
            float: The amplitude at the chosen frequency.
        """
        if margin_frequency:
            frequency_index = self._get_frequency_index(frequency)
            margin_index = int(margin_frequency / self.frequency_resolution)
            return np.mean(self.spectrum[0,
                                         frequency_index - margin_index:
                                         frequency_index + margin_index])
        else:
            return self.spectrum[0,self._get_frequency_index(frequency)]

    def get_peak_magnitude_within_window(self, window: tuple = (17,20)) -> dict:
        """Get the peak magnitude and frequency within a specific window.

        Args:
            window (tuple, optional): The window around the frequency of
                                      interest. Defaults to (17,20).

        Returns:
            dict: A dictionary containing the window, the peak magnitude and
                  the peak frequency.
        """
        index_window = (self._get_frequency_index(window[0]),
                        self._get_frequency_index(window[1]))
        magnitude_window = self.spectrum[:,index_window[0]:index_window[1]]
        peak_magnitude = np.max(magnitude_window, axis=1)
        peak_frequency_Hz = self.frequencies[
            np.argmax(magnitude_window, axis=1)
        ]

        return {
            "window_Hz": window,
            "peak_magnitude": peak_magnitude,
            "peak_frequency_Hz": peak_frequency_Hz
        }
