
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
from numpy.lib.stride_tricks import sliding_window_view

# python -m conda install -c conda-forge scipy or python -m pip install scipy
import scipy


# from EEG_quality_assessment import frequency_analysis, time_analysis
#
# class SignalMetrics:
#     """Class to store the signal metrics of an EEG signal."""
#     def __init__(self, raw: mne.io.Raw) -> None:
#         self.raw = raw
#         self.info = raw.info
#
#     def calculate_metrics(self,
#                           sliding_time_window = 1,
#                           overlap = 0.5) -> 'SignalMetrics':
#         """Calculate the signal metrics of the EEG signal."""
#
#         spectrum_object = frequency_analysis.Spectrum()
#         spectrum_object.calculate_fft(self.raw)
#         amplitude = spectrum_object.copy().calculate_amplitude()
#         amplitude._set_frequency_of_interest(18)
#         zscore = amplitude.copy().calculate_zscore()
#         snr = amplitude.copy().calculate_snr()
#         overlap = 0.5
#         data = self.raw.get_data()
#         samples = np.arange(0, data.shape[1])
#         window_nb_samples = int((sliding_time_window*self.raw.info['sfreq']))
#         step = int(window_nb_samples -window_nb_samples *overlap)
#         window_view = sliding_window_view(samples ,
#                                         window_nb_samples )[::step]
#
#         metric_names = ['average_rms',
#                         'max_gradient',
#                         'zero_crossing_rate',
#                         'hjorth_mobility',
#                         'hjorth_complexity',
#                         'kurtosis',
#                         'skewness',
#                         'variance',
#                         'signal_range',
#                         'signal_IQR']
#
#         metrics = {name: np.empty(shape = (data.shape[0], window_view.shape[0]))
#                 for name in metric_names}
#
#         for metric_name in metric_names:
#             for window_index in range(window_view.shape[0]):
#                 windowed_signal = data[:, window_view[window_index,:]]
#
#                 setattr(self, metric_name) getattr(
#                     time_analysis, metric_name)(windowed_signal)
#
#         metrics['amplitudes'] = amplitude.spectrum
#         metrics['snr'] = snr.spectrum
#         metrics['zscore'] = zscore.spectrum
