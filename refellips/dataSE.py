# -*- coding: utf-8 -*-

""""
A basic representation of a 1D dataset
"""

import numpy as np
import pandas as pd
from refnx._lib import possibly_open_file

pd.options.mode.chained_assignment = None


class DataSE(object):
    r"""
    A basic representation of a 1D dataset.

    Parameters
    ----------
    data : str, file-like or tuple of np.ndarray, optional
        String pointing to a data file.
        Alternatively it is a tuple containing the data from which the dataset
        will be constructed. The tuple should have 4 members.

            - data[0] - Wavelength (nm)
            - data[1] - Angle of incidence (degree)
            - data[2] - Psi
            - data[3] - Delta

        `data` must be four long.
        All arrays must have the same shape.

    mask : array-like
        Specifies which data points are (un)masked. Must be broadcastable
        to the data. `Data1D.mask = None` clears the mask. If a mask value
        equates to `True`, then the point is included, if a mask value equates
        to `False` it is excluded.

    reflect_delta : bool
        Specifies whether delta values are reflected around 180 degrees
        (i.e., 360 - delta[delta > 180]), as is standard for some ellipsometry
        analysis packages (i.e., WVASE).

    Attributes
    ----------
    data : tuple of np.ndarray
        The data, (wavelength, AOI, psi, delta)
    finite_data : tuple of np.ndarray
        Data points that are finite
    wavelength : np.ndarray
        wavelength (nm)
    AOI : np.ndarray
        angle of incidence (degree)
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

    def __init__(
        self, data=None, name=None, delimiter="\t", reflect_delta=False, **kwds
    ):
        self.filename = None

        self.delimiter = delimiter
        self.metadata = kwds
        self._wavelength = np.zeros(0)
        self._aoi = np.zeros(0)
        self._psi = np.zeros(0)
        self._delta = np.zeros(0)
        self.weighted = False
        self.name = name

        # If a file, then open and load the file.
        if hasattr(data, "read") or type(data) is str:
            self.load(data)
            self.filename = data

        # If already a DataSE object, then just use that.
        elif isinstance(data, DataSE):
            self.name = data.name
            self.filename = data.filename
            self.metadata = data.metadata
            self._wavelength = data._wavelength
            self._aoi = data._aoi
            self._psi = data._psi
            self._delta = data._delta

        # If a list or tuple, then assume its in format wavelength, AOI, psi, delta.
        elif isinstance(data, (list, tuple, np.ndarray)):
            self._wavelength = data[0]
            self._aoi = data[1]
            self._psi = data[2]
            self._delta = data[3]

        self._delta_flipped = False
        if reflect_delta:
            dmask = self._delta > 180
            self._delta[dmask] = 360 - self._delta[dmask]
            self._delta_flipped = True

        self.mask = np.ones_like(self._wavelength, dtype=bool)

    def __len__(self):
        """Number of unmasked points in the dataset."""
        return self.wavelength.size

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

    def unique_wavelength_data(self):
        """
        Generator yielding wavelength, AOI, psi, delta tuples for the unique
        wavelengths in a dataset (i.e. all the data points for a given
        wavelength)

        Returns
        -------
        wavelength, AOI, psi, delta
        """
        unique_wavs = np.unique(self.wavelength)
        for unique_wav in unique_wavs:
            loc = np.where(self.wavelength == unique_wav)
            yield unique_wav, self.aoi[loc], self.psi[loc], self.delta[loc]

    @property
    def wavelength(self):
        """Wavelength."""
        if self._wavelength.size > 0:
            return self._wavelength[self.mask]
        else:
            return self._wavelength

    @property
    def aoi(self):
        """Angle of incidence."""
        if self._aoi.size > 0:
            return self._aoi[self.mask]
        else:
            return self._aoi

    @property
    def psi(self):
        """Ellipsometric parameter psi."""
        if self._psi.size > 0:
            return self._psi[self.mask]
        else:
            return self._psi

    @property
    def delta(self):
        """Ellipsometric parameter delta."""
        if self._delta.size > 0:
            return self._delta[self.mask]
        else:
            return self._delta

    @property
    def data(self):
        """4-tuple containing the (lambda, AOI, psi, delta) data."""
        return self.wavelength, self.aoi, self.psi, self.delta

    @data.setter
    def data(self, data_tuple):
        """
        Set the data for this object from the supplied data.

        Parameters
        ----------
        data_tuple : tuple
            4 member tuple containing the (wav, aoi, psi, delta) data to
            specify the dataset.

        Notes
        -----
        Clears the mask for the dataset, it will need to be reapplied.

        """
        self._wavelength = np.array(data_tuple[0], dtype=float)
        self._aoi = np.array(data_tuple[1], dtype=float)
        self._psi = np.array(data_tuple[2], dtype=float)
        self._delta = np.array(data_tuple[3], dtype=float)

        self.mask = np.ones_like(self._wavelength, dtype=bool)

    def save(self, f):
        """
        Save the data to file. Saves the data as a 4 column ASCII file.

        Parameters
        ----------
        f : file-handle or string
            File to save the dataset to.

        """
        header = "wavelength\tAOI\tPsi\tDelta"
        np.savetxt(
            f,
            np.column_stack((self._wavelength, self._aoi, self._psi, self._delta)),
            delimiter="\t",
            header=header,
        )

    def load(self, f):
        """
        Load a dataset from file.
        Must be a 4 column ASCII file with columns [wavelength, AOI, Psi, Delta].

        Parameters
        ----------
        f : file-handle or string
            File to load the dataset from.

        """

        skip_lines = 0
        with possibly_open_file(f, "r") as text:
            for i in range(100):  # check the first 100 lines
                try:
                    float(text.readline().split(self.delimiter)[0])
                    break
                except ValueError:
                    skip_lines += 1

        self._wavelength, self._aoi, self._psi, self._delta = np.loadtxt(
            f, skiprows=skip_lines, delimiter=self.delimiter, encoding="utf8"
        ).T

    def refresh(self):
        """
        Refreshes a previously loaded dataset.

        """
        if self.filename is not None:
            with open(self.filename) as f:
                self.load(f)


def open_EP4file(fname, reflect_delta=False):
    """
    Open and load in an Accurion EP4 formmated data file.
    Typically a .dat file.

    Note: This file parser has been written for specific Accurion ellipsometers
    EP3 and EP4. No work has been done to ensure it is compatable with all
    Accurion ellipsometers. If you have trouble with this parser contact the
    maintainers through github.

    Parameters
    ----------
    fname : file-handle or string
        File to load the dataset from.

    reflect_delta : bool
        Option to reflect delta around 180 degrees (as WVASE would).

    Returns
    ----------
    datasets : DataSE structure
        Structure containing wavelength, angle of incidence, psi and delta.


    """
    df = pd.read_csv(fname, sep="\t", skiprows=[1])
    df = df.dropna(0, how="any")

    try:
        df["Time"]
        time_data = True
    except KeyError:
        time_data = False
        print("No time data.")

    if time_data and len(df["Time"].drop_duplicates()) > 1:
        print("Treating as time series:")
        output = []
        for t in df["Time"].drop_duplicates():
            tdf = df[df["Time"] == t]
            output += _loadEP4(tdf)  # not sure if this will work
            output[-1]["time"] = t
    else:
        output = _loadEP4(df)
        for op in output:
            op["time"] = None

    datasets = []
    for op in output:
        data = [op["lambda"], op["aoi"], op["psi"], op["delta"]]
        del op["lambda"]
        del op["aoi"]
        del op["psi"]
        del op["delta"]
        name = _make_EP4dname(fname, op)
        datasets.append(DataSE(data, name=name, reflect_delta=reflect_delta, **op))

    if len(datasets) == 1:
        return datasets[0]
    else:
        return datasets


def _make_EP4dname(name, metadata):
    """
    Create a helpful name for a data set based on an Accurion EP4
    formatted data file.

    Parameters
    ----------
    name : file-handle or string
        File name of data set.

    metadata : dict
        Dict containinng 'X pos', 'Y pos' and 'time' data.

    Returns
    ----------
    base : string
        Helpful name for the data set.

    """
    base = name[: -len("_20200929-083122.ds.dat")]
    if metadata["X pos"] is not None:
        base += f"_x={metadata['X pos']}mm_y={metadata['Y pos']}mm"
    if metadata["time"] is not None:
        base += f"_t={metadata['time']}s"

    return base


def custom_round(x, base=0.25):
    """
    Perform rounding to a particular base. Default base is 0.25.

    Parameters
    ----------
    x : DataFrame, array or list
        Data to be rounded.

    base : float
        Base that the rounding will be with respect to.

    Returns
    ----------
    Result of cutsom round : np.array

    """
    x = np.array(x, dtype=float)
    return np.round((base * np.round(x / base)), 2)


def _loadEP4(df):
    """
    Specifically loading a data file created by an Accurion EP4 ellipsometer.
    Dataframe should have colums ['#Lambda','AOI','Psi','Delta'].
    Optionally can also have columns [X_pos, Y_pos].


    Parameters
    ----------
    df : DataFrame
        Data frame containing the wavelength, angle of incidence, psi and
        delta data.

    Returns
    ----------
    output : list of dicts
        Dicts containing wavelength, angle of indcidence, psi, delta and
        possible X pos and Y pos.

    """

    try:
        df["X_pos"]
        df["Y_pos"]
        loc_data = True
    except KeyError:
        loc_data = False

    if loc_data and (
        len(df["X_pos"].drop_duplicates()) > 1 or len(df["Y_pos"].drop_duplicates()) > 1
    ):
        print("Treating as multiple locations")
        df = df[["#Lambda", "AOI", "Psi", "Delta", "X_pos", "Y_pos"]]
        df.loc[:, "X_pos"] = custom_round(df["X_pos"], base=0.25)
        df.loc[:, "Y_pos"] = custom_round(df["Y_pos"], base=0.25)

        output = []
        for x in df["X_pos"].drop_duplicates():
            for y in df["Y_pos"].drop_duplicates():
                pdf = df[np.logical_and(df["X_pos"] == x, df["Y_pos"] == y)]
                pdf = pdf[["#Lambda", "AOI", "Psi", "Delta"]]
                if len(pdf.index) > 0:
                    ave_pos = pdf.groupby(["AOI", "#Lambda"]).mean()
                    ave_pos = ave_pos.reset_index()
                    summary = {
                        "lambda": np.array(ave_pos["#Lambda"]),
                        "aoi": np.array(ave_pos["AOI"]),
                        "psi": np.array(ave_pos["Psi"]),
                        "delta": np.array(ave_pos["Delta"]),
                        "X pos": x,
                        "Y pos": y,
                    }
                    output.append(summary)
    else:
        print("Treating as single location")
        df = df[["#Lambda", "AOI", "Psi", "Delta"]]
        ave_pos = df.groupby(["AOI", "#Lambda"]).mean()
        ave_pos = ave_pos.reset_index()

        summary = {
            "lambda": np.array(ave_pos["#Lambda"]),
            "aoi": np.array(ave_pos["AOI"]),
            "psi": np.array(ave_pos["Psi"]),
            "delta": np.array(ave_pos["Delta"]),
            "X pos": None,
            "Y pos": None,
        }
        output = [summary]

    return output


def open_HORIBAfile(fname, reflect_delta=False, lambda_cutoffs=[-np.inf, np.inf]):
    """
    Opening and loading in a data file created by a Horiba ellipsometer. Data
    file loaded should be of the Horiba file format .spe.

    Note: This file parser has been written for a specific ellipsometer, no
    work has been done to ensure it is compatable with all Horiba
    ellipsometers. If you have trouble with this parser contact the maintainers
    through github.

    Parameters
    ----------
    fname : file-handle or string
        File to load the dataset from.

    reflect_delta : bool
        Option to reflect delta around 180 degrees (as WVASE would).

    lambda_cutoffs : list
        Specifies the minimum and maximum wavelengths of data to be loaded.
        List has length 2.

    Returns
    ----------
    DataSE : DataSE structure
        The data file structure from the loaded Horiba file.

    """

    name = fname[:-4]
    metadata = {}
    linenodict = {}
    MDingest = False

    with open(fname, "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            l = line[:-1]  # Drop newline character
            if not MDingest:
                if len(l) and l[0] == "#":
                    MDlabel = " ".join(l.split(" ")[1:])[:-1]
                    metadata[MDlabel] = []
                    linenodict[MDlabel] = i
                    MDingest = True

            else:
                if not len(l):
                    MDingest = False
                    if not len(metadata[MDlabel]):  # there is no metadata for entry
                        metadata[MDlabel] = None  # Set metadata to none
                    elif len(metadata[MDlabel]) == 1:  # there is only one entry
                        metadata[MDlabel] = metadata[MDlabel][
                            0
                        ]  # remove data from list

                else:  # there is metadata in the line
                    metadata[MDlabel].append(l)  # append line to metadata entry

    data_df = pd.read_csv(
        fname,
        skiprows=linenodict["DATA"] + 1,
        nrows=len(metadata["DATA"]) - 1,
        encoding="ANSI",
        delimiter=" ",
        usecols=["nm", "Psi", "Delta"],
    )

    AOI = float(metadata["INCIDENCE ANGLE"][:5])
    data_df["AOI"] = AOI * np.ones_like(data_df["nm"])
    data_df = data_df[data_df["nm"] > lambda_cutoffs[0]]
    data_df = data_df[data_df["nm"] < lambda_cutoffs[1]]

    data = [data_df["nm"], data_df["AOI"], data_df["Psi"], data_df["Delta"]]

    return DataSE(data, name=name, reflect_delta=reflect_delta, **metadata)
