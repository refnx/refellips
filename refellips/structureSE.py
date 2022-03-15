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
import numpy as np
import os
import warnings

from refnx.reflect.structure import Scatterer, Structure, Slab, Component
from refnx.analysis import Parameters, Parameter, possibly_create_parameter


class RI(Scatterer):
    """
    Object representing a materials wavelength-dependent refractive index.

    An issue is that optical parameters are supplied in units of micrometers
    ('cause thats what seems to be used in refractive index repos and
    cauchy models), the wavelength of the incident radiation is supplied in
    nanometers (thats typical) and the fitting is done in angstroms. Very
    unpleasant.

    Parameters
    ----------
    dispersion : str, {tuple, np.ndarray)
        If a string then a dispersion curve will be loaded from a file that
        the string points to. The file is assumed to be of CSV format, with the
        first column holding the wavelength (in *microns*), with the second
        column specifying the refractive index. An optional third column can be
        present that should hold the extinction coefficient.

        If  `dispersion` has length 2 (float, float), then dispersion[0] points
        to the refractive index of the material and dispersion[1] points to the
        extinction coefficient. This refractive index is assumed to be
        wavelength independent.

        If `dispersion` has length 3, then dispersion[0], dispersion[1],
        dispersion[2] are assumed to hold arrays specifying the wavelength (in
        *microns*), refractive index, and extinction coefficient.
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
    An RI object can be used to create a Slab
    """

    def __init__(
        self, dispersion=None, A=None, B=0, C=0, wavelength=658, name=""
    ):
        super(RI, self).__init__(name=name)
        self.A = None

        # attribute required by Scatterer for energy dispersive calculations
        # to work
        self.dispersive = True
        self.wavelength = wavelength

        # _wav is only set if a wavelength dependent dispersion curve is loaded
        # assumed to be in nm
        self._wav = None
        self._RI = None
        self._EC = None

        assert np.logical_xor(
            dispersion is None, A is None
        ), "Supply either values or cauchy parameters"

        if dispersion is not None:
            if type(dispersion) is str:
                if not len(name):
                    # if there is no name get it from the path
                    name = os.path.basename(dispersion).split(".")[0]

                vals = np.loadtxt(
                    dispersion, skiprows=1, delimiter=",", encoding="utf8"
                ).T
                self._wav = vals[0]
                self._RI = vals[1]
                self._EC = np.zeros_like(self._wav)
                if len(vals) == 3:
                    self._EC = vals[2]
                # convert wavelength from um to nm
                self._wav = self._wav * 1000
            elif len(dispersion) == 2:
                self._RI, self._EC = dispersion
            elif len(dispersion) == 3:
                # this is if you have an (3, N) array or tuple specifying
                # wavelength, RI, extinction coef.
                # wavelength assumed to be in *nm*
                self._wav, self._RI, self._EC = dispersion
                self._wav *= 1000
            else:
                raise TypeError("format not recognised")

        self._parameters = Parameters(name=name)

        if A is not None:
            self.A = possibly_create_parameter(A, name=f"{name} - cauchy A")
            self.B = possibly_create_parameter(B, name=f"{name} - cauchy B")
            self.C = possibly_create_parameter(C, name=f"{name} - cauchy C")
            self._parameters.extend([self.A, self.B, self.C])

    @property
    def parameters(self):
        return self._parameters

    def __str__(self):
        ri = self.complex(None)
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
            return ri_real + 1j * ri_imag

        elif self.A is not None:
            real = (
                self.A.value
                + (self.B.value * 1000**2) / (wav**2)
                + (self.C.value * 1000**4) / (wav**4)
            )
            return real + 1j * 0.0
        else:
            return self._RI + 1j * self._EC

    def __or__(self, other):
        # c = self | other
        slab = self()
        return slab | other

    def __call__(self, thick=0, rough=0, vfsolv=0):
        """
        Create a :class:`SlabSE`.

        Parameters
        ----------
        thick: refnx.analysis.Parameter or float
            Thickness of slab in Angstrom
        rough: refnx.analysis.Parameter or float
            Roughness of slab in Angstrom
        vfsolv: refnx.analysis.Parameter or float
            Volume fraction of water in slab

        Returns
        -------
        slab : SlabSE
            The newly made Slab.

        """
        return SlabSE(thick, self, rough, name=self.name, vfsolv=vfsolv)


class ComponentSE(Component):
    """
    A base class for describing the structure of a subset of an interface.

    Parameters
    ----------
    name : str, optional
        The name associated with the Component

    Notes
    -----
    By setting the `Component.interfaces` property one can control the
    type of interfacial roughness between all the layers of an interfacial
    profile.
    """

    def __or__(self, other):
        """
        OR'ing SlabSE can create a :class:`StructureSE`.

        Parameters
        ----------
        other: refellips.StructureSE, refnx.reflect.Component
            Combines with this component to make a Structure

        Returns
        -------
        s: refellips.StructureSE
            The created Structure

        """
        # c = self | other
        p = StructureSE()
        p |= self
        p |= other
        return p

    def __mul__(self, n):
        """
        MUL'ing components makes them repeat.

        Parameters
        ----------
        n: int
            How many times you want to repeat the Component

        Returns
        -------
        s: refellipse.StructureSE
            The created Structure
        """
        # convert to integer, should raise an error if there's a problem
        n = operator.index(n)
        if n < 1:
            return StructureSE()
        elif n == 1:
            return self
        else:
            s = StructureSE()
            s.extend([self] * n)
            return s


class SlabSE(ComponentSE, Slab):
    """
    A slab component has uniform RI over its thickness.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld : :class:`refnx.reflect.Scatterer`, complex, or float
        (complex) refractive index of film
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).
    """


class StructureSE(Structure):
    """
    Represents the interfacial Structure of an Ellipsometry sample.
    Successive Components are added to the Structure to construct the
    interface.
    """
    # the only reason to have this class is to have the correct overall_sld
    # method
    def __init__(self, *args, ema='linear', **kwds):
        super().__init__(*args, **kwds)
        self.ema = ema

    def __ior__(self, other):
        """
        Build a structure by `IOR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`StructureSE`, :class:`ComponentSE`, :class:`RI`
            The object to add to the structure.
        """
        # self |= other
        if isinstance(other, ComponentSE):
            self.append(other)
        elif isinstance(other, StructureSE):
            self.extend(other.data)
        elif isinstance(other, Scatterer):
            slab = other(0, 0)
            self.append(slab)
        else:
            raise ValueError()

        return self

    def __or__(self, other):
        """
        Build a structure by `OR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`StructureSE`, :class:`ComponentSE`, :class:`SLD`
            The object to add to the structure.
        """
        # c = self | other
        p = StructureSE()
        p |= self
        p |= other
        return p

    def overall_sld(self, slabs, solvent):
        """
        Calculates the overall refractive index of the material and solvent RI
        in a layer.

        Parameters
        ----------
        slabs : np.ndarray
            Slab representation of the layers to be averaged.
        solvent : complex or RI
            RI of solvating material.

        Returns
        -------
        averaged_slabs : np.ndarray
            the averaged slabs.

        Notes
        -----
        This method is called `overall_sld` to take advantage of inheritance, i.e.
        we don't have to override the `slabs` method. It definitely does calculate
        the overall RI.
        """
        solv = solvent
        if isinstance(solvent, Scatterer):
            solv = solvent.complex(self.wavelength)

        return overall_RI(slabs, solv, ema=self.ema)


def overall_RI(slabs, solvent, ema='linear'):
    """
    Calculates the overall refractive index of the material and solvent RI
    in a layer.

    Parameters
    ----------
    slabs : np.ndarray
        Slab representation of the layers to be averaged.
    solvent : complex or RI
        RI of solvating material.
    ema : {'linear'}
        Specifies how refractive indices are mixed together

    Returns
    -------
    averaged_slabs : np.ndarray
        the averaged slabs.
    """
    if ema == 'linear':
        slabs[..., 1:3] = slabs[..., 1:3]**2
        slabs[..., 1:3] *= (1 - slabs[..., 4])[..., np.newaxis]

        slabs[..., 1] += solvent.real**2 * slabs[..., 4]
        slabs[..., 2] += solvent.imag**2 * slabs[..., 4]
        slabs[..., 1:3] = np.sqrt(slabs[..., 1:3])
    else:
        raise RuntimeError("No other method of mixing is known")

    return slabs
