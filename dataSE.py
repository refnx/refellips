# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 08:11:49 2020

@author: Isaac
"""
""""
A basic representation of a 1D dataset
"""
import os.path
import re

import numpy as np
from scipy._lib._util import check_random_state
from refnx.util.nsplice import get_scaling_in_overlap
from refnx._lib import possibly_open_file


class DataSE(object):
    r"""
    A basic representation of a 1D dataset.

    Parameters
    ----------
    data : str, file-like or tuple of np.ndarray, optional
        `data` can be a string or file-like object referring to a File to load
        the dataset from. The file should be plain text and have 2 to 4
        columns separated by space, comma or tab. The columns represent
        `x, y [y_err [, x_err]]`.

        Alternatively it is a tuple containing the data from which the dataset
        will be constructed. The tuple should have between 2 and 4 members.

            - data[0] - x
            - data[1] - y
            - data[2] - uncertainties on y, y_err
            - data[3] - uncertainties on x, x_err

        `data` must be at least two long, `x` and `y`.
        If the tuple is at least 3 long then the third member is `y_err`.
        If the tuple is 4 long then the fourth member is `x_err`.
        All arrays must have the same shape.

    mask : array-like
        Specifies which data points are (un)masked. Must be broadcastable
        to the y-data. `Data1D.mask = None` clears the mask. If a mask value
        equates to `True`, then the point is included, if a mask value equates
        to `False` it is excluded.

    Attributes
    ----------
    data : tuple of np.ndarray
        The data, (x, y, y_err, x_err)
    finite_data : tuple of np.ndarray
        Data points that are finite
    wav : np.ndarray
        wavelength
    AOI : np.ndarray
        angle of incidence
    psi : np.ndarray
        psi
    delta : np.ndarray
        delta
    mask : np.ndarray
        mask
    filename : str or None
        The file the data was read from
    weighted : bool
        Whether the y data has uncertainties
    metadata : dict
        Information that should be retained with the dataset.

    """

    def __init__(self, data=None, **kwds):
        self.filename = None
        self.name = None

        self.metadata = kwds
        self._wav = np.zeros(0)
        self._aoi = np.zeros(0)
        self._psi = np.zeros(0)
        self._delta = np.zeros(0)
        self.weighted = False


        # if it's a file then open and load the file.
        if hasattr(data, "read") or type(data) is str:
            self.load(data)
            self.filename = data
        elif isinstance(data, DataSE):
            self.name = data.name
            self.filename = data.filename
            self.metadata = data.metadata
            self._wav = data._wav
            self._aoi = data._aoi
            self._psi = data._psi
            self._delta = data._delta

        self.mask = np.ones_like(self._wav, dtype=bool)

    def __len__(self):
        """Number of unmasked points in the dataset."""
        return self.wav.size

    def __str__(self):
        return "<{0}>, {1} points".format(self.name, len(self))

    def __repr__(self):
        msk = self.mask
        if np.all(self.mask):
            msk = None

        d = {"filename": self.filename, "msk": msk, "data": self.data}
        if self.filename is not None:
            return "Data1D(data={filename!r}," " mask={msk!r})".format(**d)
        else:
            return "Data1D(data={data!r}," " mask={msk!r})".format(**d)

    def __iter__(self):
        self._current_wav_idx = 0
        self._current_wav = self._unique_wavs[self._current_wav_idx]
        return self

    def __next__(self):
        if self._current_wav_idx < len(self._unique_wavs):
            self._current_wav = self._unique_wavs[self._current_wav_idx]
            self.mask = self._wav == self._current_wav
            self._current_wav_idx += 1
            return self
        else:
            raise StopIteration

    @property
    def _unique_wavs(self):
        """
        List of wavelengths in dataset.

        Returns
        -------
        np.array
            contains wavelengths used by dataset

        """
        return np.unique(self.wav)

    @property
    def wav(self):
        """Wavelength."""
        if self._wav.size > 0:
            return self._wav[self.mask]
        else:
            return self._wav

    @property
    def aoi(self):
        """Angle of incidence."""
        if self._aoi.size > 0:
            return self._aoi[self.mask]
        else:
            return self._aoi

    @property
    def psi(self):
        """Angle of incidence."""
        if self._psi.size > 0:
            return self._psi[self.mask]
        else:
            return self._psi

    @property
    def delta(self):
        """Angle of incidence."""
        if self._delta.size > 0:
            return self._delta[self.mask]
        else:
            return self._delta

    @property
    def data(self):
        """4-tuple containing the (lambda, AOI, delta, psi) data."""
        return self.wav, self.aoi, self.delta, self.psi


    @data.setter
    def data(self, data_tuple):
        """
        Set the data for this object from supplied data.

        Parameters
        ----------
        data_tuple : tuple
            4 member tuple containing the (wav, aoi, psi, delta) data to
            specify the dataset.

        Notes
        -----
        Clears the mask for the dataset, it will need to be reapplied.

        """
        self._wav = np.array(data_tuple[0], dtype=float)
        self._aoi = np.array(data_tuple[1], dtype=float)
        self._psi = np.array(data_tuple[2], dtype=float)
        self._delta = np.array(data_tuple[3], dtype=float)

        self.mask = np.ones_like(self._wav, dtype=bool)
        
    

    def save(self, f):
        """
        Save the data to file. Saves the data as 4 column ASCII.

        Parameters
        ----------
        f : file-handle or string
            File to save the dataset to.

        """
        header = 'wavelength\tAOI\tPsi\tDelta'
        np.savetxt(
            f, np.column_stack((self._wav, self._aoi, self._psi, self._delta)),
            delimiter='\t', header=header
        )

    def load(self, f):
        """
        Load a dataset from file. Must be 4 column ASCII.
        
        wavelength, AOI, Psi, Delta

        Parameters
        ----------
        f : file-handle or string
            File to load the dataset from.

        """
        # The problem here is that there is no standard ellipsometry file

        skip_lines = 0
        with open(f, 'r') as text:
            for i in range(100): # check the first 100 lines
                try:
                    float(text.readline().split('\t')[0])
                    break
                except ValueError:
                    skip_lines += 1

        self._wav, self._aoi, self._psi, self._delta = np.loadtxt(f, skiprows=skip_lines).T


    def refresh(self):
        """
        Refreshes a previously loaded dataset.

        """
        if self.filename is not None:
            with open(self.filename) as f:
                self.load(f)