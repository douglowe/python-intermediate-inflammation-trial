"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: A 2D data array (each row should contain the data for one day)
    :returns: an array of mean values for each day
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array."""
    return np.min(data, axis=0)


# TODO(lesson-design) Add Patient class
# TODO(lesson-design) Implement data persistence
# TODO(lesson-design) Add Doctor class

def patient_normalise(data):
    """Normalise patient data from a 2D inflammation data array."""
    if np.any(data <0):
        raise ValueError('Inflammation values should not be negative')
    max = np.max(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised