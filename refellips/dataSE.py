# -*- coding: utf-8 -*-

""""
A basic representation of a 1D dataset
"""

import copy
import numpy as np
import pandas as pd
from refnx._lib import possibly_open_file
from pathlib import PurePath

pd.options.mode.chained_assignment = None


class DataSE(object):
    r"""
    A basic representation of a 1D dataset.

    Parameters
    ----------
    data : {str, file-like, Path, tuple of np.ndarray}, optional
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
    AOI : np.ndarray
        angle of incidence (degree)
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
        # TODO when we come up with measurement uncertainties change this.
        self.weighted = False
        self.name = name

        # If a file, then open and load the file.
        if (
            hasattr(data, "read")
            or type(data) is str
            or isinstance(data, PurePath)
        ):
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
        """wavelength(nm)"""

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
        """4-tuple containing the (wavelength), AOI, psi, delta) data."""
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
            np.column_stack(
                (self._wavelength, self._aoi, self._psi, self._delta)
            ),
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
    EP3 and EP4. No work has been done to ensure it is compatible with all
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
    df = df.dropna(axis=0, how="any")
    # normally the NaN are at the end of the file, but they can also be in
    # the middle
    df = df.reset_index()

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
        datasets.append(
            DataSE(data, name=name, reflect_delta=reflect_delta, **op)
        )

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
    name = str(name)
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
        len(df["X_pos"].drop_duplicates()) > 1
        or len(df["Y_pos"].drop_duplicates()) > 1
    ):
        xpos = np.nan
        ypos = np.nan

        area_indices = []
        for entry in df.iterrows():
            if (not np.allclose(xpos, entry[1]["X_pos"], atol=0.2)) or (
                not np.allclose(ypos, entry[1]["Y_pos"], atol=0.2)
            ):
                idx = entry[0]
                xpos = entry[1]["X_pos"]
                ypos = entry[1]["Y_pos"]
                area_indices.append(idx)
        area_indices.append(len(df))

        if len(area_indices) > 2:
            print("Treating as multiple locations")
        else:
            print("Treating as single location")

        output = []
        for i in range(len(area_indices) - 1):
            pdf = df.loc[area_indices[i] : area_indices[i + 1] - 1][
                ["#Lambda", "AOI", "Psi", "Delta", "X_pos", "Y_pos"]
            ]

            if len(pdf.index) > 0:
                ave_pos = pdf.groupby(["AOI", "#Lambda"]).mean()
                ave_pos = ave_pos.reset_index()

                summary = {
                    "lambda": np.array(ave_pos["#Lambda"]),
                    "aoi": np.array(ave_pos["AOI"]),
                    "psi": np.array(ave_pos["Psi"]),
                    "delta": np.array(ave_pos["Delta"]),
                    "X pos": np.round(np.mean(ave_pos["X_pos"]), 2),
                    "Y pos": np.round(np.mean(ave_pos["Y_pos"]), 2),
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


def open_HORIBAfile(
    fname, reflect_delta=False, lambda_cutoffs=[-np.inf, np.inf]
):
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
            line = line.strip()  # Drop newline character
            if not MDingest:
                if len(line) and line[0] == "#":
                    MDlabel = " ".join(line.split(" ")[1:])[:-1]
                    metadata[MDlabel] = []
                    linenodict[MDlabel] = i
                    MDingest = True

            else:
                if not len(line):
                    MDingest = False
                    if not len(
                        metadata[MDlabel]
                    ):  # there is no metadata for entry
                        metadata[MDlabel] = None  # Set metadata to none
                    elif len(metadata[MDlabel]) == 1:  # there is only one entry
                        metadata[MDlabel] = metadata[MDlabel][
                            0
                        ]  # remove data from list

                else:  # there is metadata in the line
                    metadata[MDlabel].append(
                        line
                    )  # append line to metadata entry

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


def open_M2000file(fname, dropdatapoints=1):
    """
    Open and load in an Accurion EP4 formmated data file.
    Typically a .dat file.

    Note: This file parser has been written for specific Accurion ellipsometers
    EP3 and EP4. No work has been done to ensure it is compatible with all
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

    data = []

    with open(fname, mode="r") as file:
        __ = file.readline()
        meas_info = file.readline()
        __ = file.readline()

        count = 0
        while True:
            data_row = []

            count += 1
            # print (count)

            # Get next line from file
            line = file.readline().split("\t")
            if not line:
                break
            if len(line) == 1:
                break
            data_row.append(float(line[0]))  # Wavelength
            data_row.append(float(line[1]))  # Angle
            data_row.append(float(line[2]))  # Psi
            data_row.append(float(line[3]))  # Delta
            data_row.append(float(line[4]))  # Psi Error
            data_row.append(float(line[5]))  # Delta Error

            line = file.readline().split("\t")
            data_row.append(float(line[2]))  # Unknown
            data_row.append(float(line[3]))  # Depolarization %
            data_row.append(float(line[4]))  # Unknown

            line = file.readline().split("\t")
            data_row.append(float(line[2]))  # Unknown
            data_row.append(float(line[3]))  # Intensity
            data_row.append(float(line[4]))  # Unknown

            data.append(data_row)

    data = np.array(data)
    data = data[::dropdatapoints]
    return DataSE(data[:, [0, 1, 2, 3]].T)


def open_woolam_time_series(fname, take_every=1):
    df = pd.read_csv(
        fname,
        skiprows=4,
        sep="\t",
        names=[
            "Wavelength, nm",
            "Angle of incidence, ˚",
            "Psi",
            "Delta",
            "Psi error",
            "Delta error",
            "None",
            "Time, min",
        ],
    )

    time_dict = {}
    for idx, (time, subdf) in enumerate(df.groupby("Time, min")):
        if idx % take_every == 0:
            time_dict[np.round(time * 60, 1)] = DataSE(
                np.array(
                    [
                        subdf["Wavelength, nm"],
                        subdf["Angle of incidence, ˚"],
                        subdf["Psi"],
                        subdf["Delta"],
                    ]
                )[:, ::5]
            )

    return time_dict


def open_FilmSenseFile(fname):
    with open(fname, "r") as f:
        header = f.readline()
        if header == "Film_Sense_Data\n":
            return _open_FilmSenseFile_standard(f)
        elif header == "Film_Sense_Dyn_Data\n":
            return _open_FilmSenseFile_dynamic(f)
        else:
            assert False, "Filetype not recognized"


def _parse_FilmSenseFileHeader(firstline, mode="standard"):
    firstline = firstline[:-1]  # remove newline char
    firstline = firstline.split("\t")

    metadata = {
        "numwvls": int(firstline[0]),
        "numdatasets": int(firstline[1]),
        "nomAOI": float(firstline[2]),
    }

    if mode == "standard":
        metadata["AlignX"] = float(firstline[3])
        metadata["AlignY"] = float(firstline[4])
        metadata["AvInten"] = float(firstline[5])
    elif mode == "dynamic":
        metadata["?"] = float(firstline[3])
    else:
        assert False, "mode not recognized"

    return metadata


def _open_FilmSenseFile_standard(f):
    metadata = _parse_FilmSenseFileHeader(f.readline())

    # Note - in the documentation the first numwvls lines are only supposed
    # have 4 columns. In these data files they have 8.

    df = pd.DataFrame(
        columns=[
            "Wavelength",
            "led_Br",
            "led_ExpL",
            "led_ExpR",
            "N",
            "C",
            "S",
            "P",
            "Intensity",
            "Delta",
            "Psi",
        ],
        index=np.linspace(
            1, metadata["numwvls"], metadata["numwvls"], dtype=int
        ),
    )

    for i in range(metadata["numwvls"]):
        line = f.readline().split("\t")
        df.iloc[i]["Wavelength"] = float(line[0])
        df.iloc[i]["led_Br"] = float(line[1])
        df.iloc[i]["led_ExpL"] = float(line[2])
        df.iloc[i]["led_ExpR"] = float(line[3])

    for i in range(metadata["numwvls"]):
        line = f.readline().split("\t")
        df.iloc[i]["N"] = float(line[0])
        df.iloc[i]["C"] = float(line[1])
        df.iloc[i]["S"] = float(line[2])
        df.iloc[i]["P"] = float(line[3])
        df.iloc[i]["Intensity"] = float(line[4])

    S = np.array(df["S"], dtype=np.float32)
    N = np.array(df["N"], dtype=np.float32)
    C = np.array(df["C"], dtype=np.float32)
    df["Psi"] = np.rad2deg(np.arccos(N) / 2)
    # Delta1 = 180+np.rad2deg(np.arctan(np.array(df['S'], dtype=np.float32)/np.array(df['C'], dtype=np.float32)))

    df["Delta"] = np.rad2deg(np.angle((C + 1j * S) / (1 + N)))

    Psi = np.array(df["Psi"]).astype(np.float64)
    Delta = np.array(df["Delta"]).astype(np.float64)

    Deltamask = Delta < 0
    Delta[Deltamask] = 360 + Delta[Deltamask]

    AOI = np.ones_like(Psi) * metadata["nomAOI"]
    Wvl = np.array(df["Wavelength"]).astype(np.float64)

    return DataSE(data=[Wvl, AOI, Psi, Delta], reflect_delta=False)


def _open_FilmSenseFile_dynamic(f):
    metadata = _parse_FilmSenseFileHeader(f.readline(), mode="dynamic")

    base_df = pd.DataFrame(
        columns=[
            "Wavelength",
            "led_Br",
            "led_ExpL",
            "led_ExpR",
            "N",
            "C",
            "S",
            "P",
            "Intensity",
            "Delta",
            "Psi",
        ],
        index=np.linspace(
            1, metadata["numwvls"], metadata["numwvls"], dtype=int
        ),
    )

    for i in range(metadata["numwvls"]):
        line = f.readline().split("\t")
        base_df.iloc[i]["Wavelength"] = float(line[0])
        base_df.iloc[i]["led_Br"] = float(line[1])
        base_df.iloc[i]["led_ExpL"] = float(line[2])
        base_df.iloc[i]["led_ExpR"] = float(line[3])

    dataheader = f.readline().split("\t")

    time_series = {}
    for i in range(metadata["numdatasets"]):
        line = f.readline()[:-2]
        line = line.split("\t")
        time = float(line[0])
        df = copy.deepcopy(base_df)

        for j in range(metadata["numwvls"]):
            J = j * 5
            df.iloc[j]["N"] = float(line[J + 1])
            df.iloc[j]["C"] = float(line[J + 2])
            df.iloc[j]["S"] = float(line[J + 3])
            df.iloc[j]["P"] = float(line[J + 4])
            df.iloc[j]["Intensity"] = float(line[J + 5])

        S = np.array(df["S"], dtype=np.float32)
        N = np.array(df["N"], dtype=np.float32)
        C = np.array(df["C"], dtype=np.float32)
        df["Psi"] = np.rad2deg(np.arccos(N) / 2)
        df["Delta"] = np.rad2deg(np.angle((C + 1j * S) / (1 + N)))

        Psi = np.array(df["Psi"]).astype(np.float64)
        Delta = np.array(df["Delta"]).astype(np.float64)
        Deltamask = Delta < 0
        Delta[Deltamask] = 360 + Delta[Deltamask]
        AOI = np.ones_like(Psi) * metadata["nomAOI"]
        Wvl = np.array(df["Wavelength"]).astype(np.float64)

        time_series[time] = DataSE(
            data=[Wvl, AOI, Psi, Delta], reflect_delta=False
        )

    return time_series
