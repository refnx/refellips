""""""
"""
BSD 3-Clause License

Copyright (c) 2020-2022,
Hayden Robertson (University of Newcastle)
Isaac Gresham (University of Sydney)
Andrew Nelson (ANSTO)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# -*- coding: utf-8 -*-

from refnx.reflect.structure import Scatterer
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
import numpy as np
import os
import warnings


class RI(Scatterer):
    """
    Object representing a materials wavelength-dependent refractive index.

    An issue is that optical parameters are supplied in units of micro
    meters ('cause thats what seems to be used in refractive index repos and
    cauchy models), the wavelength of the incident radiation is supplied in
    nanometers (thats typical) and the fitting is done in angstroms. Very
    unpleasant.

    Parameters
    ----------
    value : tuple, string
    A : float or parameter
        Cauchy parameter A. If not none RI will use the cauchy model.
        Default None.
    B : float or parameter
        Cauchy parameter B in um^2. Default 0.
    C : float or parameter
        Cauchy parameter C in um^4. Default 0.
    wavelength : float
        default wavelength for calculation (nm)
    name : str, optional
        Name of material.

    Notes
    -----
    An RI object can be used to create a Slab:
    """

    def __init__(self, value=None, A=None, B=0, C=0, wavelength=658, name=""):
        if (
            type(value) is str and name == ""
        ):  # if there is no name get it from the path
            name = os.path.basename(value).split(".")[0]

        super(RI, self).__init__(name=name)
        self.A = None

        # attribute required by Scatterer for energy dispersive calculations
        # to work
        self.dispersive = True
        self.wavelength = wavelength

        # _wav is only set if a wavelength dependent dispersion curve is loaded
        # assumed to be in nm
        self._wav = None

        assert np.logical_xor(
            value is None, A is None
        ), "Supply either values or cauchy parameters"

        if value is not None:
            if type(value) is str:
                try:
                    self._wav, self._RI, self._EC = np.loadtxt(
                        value, skiprows=1, delimiter=",", encoding="utf8"
                    ).T
                except ValueError:
                    self._wav, self._RI = np.loadtxt(
                        value,
                        skiprows=1,
                        delimiter=",",
                        usecols=[0, 1],
                        encoding="utf8",
                    ).T
                    self._EC = np.zeros_like(self._wav)
                # convert wavelength from um to nm
                self._wav = self._wav * 1000
            elif len(value) == 2:
                self._RI, self._EC = value
            elif len(value) == 3:
                # this is if you have an (3, N) array or tuple specifying
                # wavelength, RI, extinction coef.
                # wavelength assumed to be in *nm*
                self._wav, self._RI, self._EC = value
                self._wav *= 1000
            else:
                raise TypeError("format not recognised")
        else:
            self.wavelength = wavelength
            self._RI = None
            self._EC = None

        self._parameters = Parameters(name=name)

        if A is not None:
            self.A = possibly_create_parameter(A, name=f"{name} - cauchy A")
            self.B = possibly_create_parameter(B, name=f"{name} - cauchy B")
            self.C = possibly_create_parameter(C, name=f"{name} - cauchy C")
            self._parameters.extend([self.A, self.B, self.C])

        # TODO test whether self.A or self._wav are None. If so, then raise
        # Exception

    @property
    def parameters(self):
        return self._parameters

    def __repr__(self):
        ri = complex(self)
        return str(f"n: {ri.real}, k: {ri.imag}")

    def __complex__(self):
        """
        The refractive index and extinction coefficient
        """
        return self.complex(None)

    def complex(self, wavelength):
        """
        Calculate a complex RI

        Parameters
        ----------
        wavelength : float
            wavelength of light in nm

        Returns
        -------
        RI : complex
            refractive index and extinction coefficient
        """
        # just in case wavelength is None
        wav = wavelength or self.wavelength

        if np.any(self._wav):
            # return a wavelength from a dispersion curve
            # TODO - raise a warning if the wavelength supplied is outside the
            # wavelength range covered by the data file.
            ri_real = np.interp(wav, self._wav, self._RI)
            ri_imag = np.interp(wav, self._wav, self._EC)
            return ri_real + 1J*ri_imag

        elif self.A is not None:
            # TODO query about the cauchy value calculation
            real = (
                self.A.value
                + (self.B.value * 1000**2) / (wav**2)
                + (self.C.value * 1000**4) / (wav**4)
            )
            return real + 1J*0.0
        else:
            return complex(self._RI, self._EC)
