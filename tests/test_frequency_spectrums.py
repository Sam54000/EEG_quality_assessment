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
from EEG_quality_assessment import frequency_spectrums as spectrums_package
import mne
import numpy
import pytest


@pytest.fixture
def raw() -> mne.io.Raw:
    """Create a raw object from mne.io.Raw."""
    return mne.io.read_raw_edf(
        './EEG_quality_assessment/tests/sub-01_ses-01_task-rest_run-01_eeg.edf'
        )

@pytest.fixture
def fft_spectrum(raw:mne.io.Raw) -> 'spectrums_package.FourierSpectrum':
    """Create a FourierSpectrum object from mne.io.Raw."""
    r = raw
    fft_object = spectrums_package.FourierSpectrum()
    fft = fft_object.calculate_fft(r)
    return fft

@pytest.fixture
def amplitude_spectrum(
    fft_spectrum:spectrums_package.FourierSpectrum
    ) -> 'spectrums_package.FourierSpectrum':
    """Create a FourierSpectrum object from mne.io.Raw."""
    return fft_spectrum.calculate_amplitude()

@pytest.fixture
def zscore_spectrum(
    amplitude_spectrum:spectrums_package.AmplitudeSpectrum
    ) -> 'spectrums_package.FourierSpectrum':
    """Create a FourierSpectrum object from mne.io.Raw."""
    return  amplitude_spectrum.calculate_zscore()

@pytest.fixture
def snr_spectrum(
    amplitude_spectrum:spectrums_package.AmplitudeSpectrum
    ) -> 'spectrums_package.FourierSpectrum':
    """Create a FourierSpectrum object from mne.io.Raw."""
    return amplitude_spectrum.calculate_snr()

def test_shape_fft(raw: mne.io.Raw,fft_spectrum:numpy.ndarray) -> None:
    """Test that fft() returns the correct value for valid input."""
    raw_data_shape = (raw.get_data().shape[0],
                      raw.get_data().shape[1] // 2)
    fft_spectrum_shape = fft_spectrum.spectrum.shape
    assert raw_data_shape == fft_spectrum_shape

def test_type_fft(fft_spectrum:numpy.ndarray) -> None:
    """Test that fft() returns the correct value for valid input."""
    assert fft_spectrum.spectrum.dtype == 'complex128'

def test_shape_frequencies(fft_spectrum:numpy.ndarray) -> None:
    assert fft_spectrum.frequencies.shape[0] == fft_spectrum.spectrum.shape[1]

def test_type_frequencies(fft_spectrum:numpy.ndarray) -> None:
    fft_spectrum._set_frequency_of_interest(13)
    desired_frequency = fft_spectrum.frequency_of_interest
    frequency_index = fft_spectrum._get_frequency_index()
    actual_frequency = fft_spectrum.frequencies[frequency_index]
    spectrum_frequency_resolution = fft_spectrum.frequency_resolution
    freq_difference = numpy.abs(actual_frequency - desired_frequency)
    assert freq_difference < spectrum_frequency_resolution
