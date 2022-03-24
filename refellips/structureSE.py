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
import os.path
import warnings
import glob

from refnx.reflect.structure import (
    Scatterer,
    Structure,
    Slab,
    Component,
    sld_profile,
)
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
from refnx._lib import flatten
from refnx.reflect import _reflect as refcalc

# contracting the SLD profile can greatly speed a reflectivity calculation up.
contract_by_area = refcalc._contract_by_area


# list of material dispersion curves distributed with refellips
_pth = os.path.dirname(os.path.abspath(__file__))
_material_files = glob.glob(os.path.join(_pth, "materials/*.csv"))
materials = [os.path.basename(m)[:-4] for m in _material_files]


class ScattererSE(Scatterer):
    """
    Abstract base class for something that will have a refractive index.
    Inherited from refnx.reflect.structure.Scatterer
    """

    def __init__(self, name="", wavelength=None):
        self.name = name
        # by default energy dispersive scatterers for ellipsometry are energy
        # dispersive
        self.dispersive = True
        self.wavelength = wavelength

    def __str__(self):
        ri = complex(self)
        return f"n: {ri.real}, k: {ri.imag}"

    def __complex__(self):
        """
        The refractive index and extinction coefficient
        """
        return self.complex(None)

    def complex(self, wavelength):
        raise NotImplementedError(
            "The complex method is not implemented for this subclass"
        )

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
        Example
        --------
        >>> # a RI object representing Silicon Dioxide using a given refractive
            index (e.g., n = 1.46) and extinction coefficent (e.g., k = 0).
        >>> sio2 = RI(dispersion=[1.46, 0], name='SiO2')
        >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
        >>> sio2_layer = sio2(20, 3)
        """
        return SlabSE(thick, self, rough, name=self.name, vfsolv=vfsolv)


class RI(ScattererSE):
    """
    Object representing a materials wavelength-dependent refractive index.

    An issue is that optical parameters are supplied in units of micrometers
    ('cause thats what seems to be used in refractive index repos and
    cauchy models), the wavelength of the incident radiation is supplied in
    nanometers (thats typical) and the fitting is done in angstroms. Very
    unpleasant.

    Parameters
    ----------
    dispersion : {str, tuple, np.ndarray)
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
    wavelength : float
        default wavelength for calculation (nm)
    name : str, optional
        Name of material.

    Notes
    -----
    An RI object can be used to create a Slab
    """

    def __init__(self, dispersion=None, wavelength=658, name=""):
        super(RI, self).__init__(name=name, wavelength=wavelength)

        # _wav is only set if a wavelength dependent dispersion curve is loaded
        # assumed to be in nm
        self._wav = None
        self._RI = None
        self._EC = None

        if dispersion is None:
            raise RuntimeError("dispersion must be specified")

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

    @property
    def parameters(self):
        return Parameters(name=self.name)

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
        wav = self.wavelength
        if np.any(wavelength):
            wav = wavelength

        if np.any(self._wav):
            # return a wavelength from a dispersion curve
            # TODO - raise a warning if the wavelength supplied is outside the
            # wavelength range covered by the data file.
            ri_real = np.interp(wav, self._wav, self._RI)
            ri_imag = np.interp(wav, self._wav, self._EC)
            return ri_real + 1j * ri_imag

        else:
            return self._RI + 1j * self._EC


def load_material(material):
    """
    Loads a dispersion curve from a file distributed with refellips.

    Parameters
    ----------
    material: str
        One of the materials in ``refellips.materials``

    Returns
    -------
    ri: refellips.RI

    Notes
    -----
    To get a list of the dispersion curves distributed with refellips examine
    the entries in ``refellips.materials``.
    """
    if material in materials:
        pth = os.path.join(_pth, "materials", f"{material}.csv")
        return RI(dispersion=pth)
    raise ValueError(f"{material} is not in the list of dispersion curves")


class Cauchy(ScattererSE):
    """
    Cauchy model for wavelength-dependent refractive index.

    Optical parameters are supplied in units of micrometers
    ('cause thats what seems to be used in refractive index repos and
    cauchy models), the wavelength of the incident radiation is supplied in
    nanometers (that's typical) and the fitting is done in angstroms.

    The refractive index is calculated as:
    ``A + (B * 1000**2) / (wav**2) + (C * 1000**4) / (wav**4)``

    where the factors of 1000 convert from microns to nm.

    Parameters
    ----------
    A : float or parameter
        Cauchy parameter A.
    B : float or parameter
        Cauchy parameter B in um^2. Default 0.
    C : float or parameter
        Cauchy parameter C in um^4. Default 0.
    wavelength : float
        default wavelength for calculation (nm)
    name : str, optional
        Name of material.
    """

    def __init__(self, A, B=0, C=0, wavelength=658, name=""):
        super().__init__(name=name, wavelength=wavelength)
        self.A = possibly_create_parameter(A, name=f"{name} - cauchy A")
        self.B = possibly_create_parameter(B, name=f"{name} - cauchy B")
        self.C = possibly_create_parameter(C, name=f"{name} - cauchy C")
        self._parameters = Parameters(name=name)
        self._parameters.extend([self.A, self.B, self.C])

    @property
    def parameters(self):
        return self._parameters

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
        wav = self.wavelength
        if np.any(wavelength):
            wav = wavelength

        real = (
            self.A.value
            + (self.B.value * 1000**2) / (wav**2)
            + (self.C.value * 1000**4) / (wav**4)
        )
        return real + 1j * 0.0


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
        other: refellips.StructureSE, refellips.ComponentSE
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


class SlabSE(ComponentSE):
    """
    A slab component has uniform SLD over its thickness
    from refnx.reflect.structure.Slab
    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    ri : :class:`refellips.ScattererSE`, complex, or float
        (complex) RI of film
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

    def __init__(self, thick, ri, rough, name="", vfsolv=0, interface=None):
        self.name = name
        self._interfaces = None
        self.thick = possibly_create_parameter(
            thick, name=f"{name} - thick", units="Å"
        )
        if isinstance(ri, ScattererSE):
            self.ri = ri
        else:
            self.ri = RI(ri)
        self.rough = possibly_create_parameter(
            rough, name=f"{name} - rough", units="Å"
        )
        self.vfsolv = possibly_create_parameter(
            vfsolv, name=f"{name} - volfrac solvent", bounds=(0.0, 1.0)
        )

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.ri.parameters)
        p.extend([self.rough, self.vfsolv])

        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"SlabSE({self.thick!r}, {self.ri!r}, {self.rough!r},"
            f" name={self.name!r}, vfsolv={self.vfsolv!r},"
            f" interface={self.interfaces!r})"
        )

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component
        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        # speculative shortcut to prevent a number of attribute retrievals
        if self.ri.dispersive:
            ric = self.ri.complex(getattr(structure, "wavelength", None))
        else:
            ric = complex(self.ri)

        return np.array(
            [
                [
                    self.thick.value,
                    ric.real,
                    ric.imag,
                    self.rough.value,
                    self.vfsolv.value,
                ]
            ],
            dtype=float,
        )


class MixedSlabSE(ComponentSE):
    """
    A slab component made of two materials.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    ri_A : ScattererSE
        refractive index of first material
    ri_B : ScattererSE
        refractive index of second material
    vf_B : float
        volume fraction of B in the layer. Volume fraction of A is calculated
        as ``1 - vf_B``.
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).
    """

    def __init__(
        self,
        thick,
        ri_A,
        ri_B,
        vf_B,
        rough,
        name="",
        interface=None,
    ):
        super().__init__(name=name)
        self.thick = possibly_create_parameter(
            thick, name=f"{name} - thick", units="Å"
        )
        self.ri_A = ri_A
        self.ri_B = ri_B
        self.vf_B = possibly_create_parameter(
            vf_B, name=f"{name} - vf_B", bounds=(0, 1)
        )

        self.rough = possibly_create_parameter(
            rough, name=f"{name} - rough", units="Å"
        )

        p = Parameters(name=self.name)
        p.append(self.thick)
        p.extend(self.ri_A.parameters)
        p.extend(self.ri_B.parameters)
        p.append(self.vf_B)
        p.append(self.rough)
        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"MixedSlabSE({self.thick!r}, {self.ri_A!r}, {self.ri_B!r},"
            f" {self.vf_B!r}, {self.rough!r}, name={self.name!r},"
            f" interface={self.interfaces!r})"
        )

    def __str__(self):
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        wavelength = getattr(structure, "wavelength", None)
        if self.ri_A.dispersive:
            riac = self.ri_A.complex(wavelength)
        else:
            riac = complex(self.ri_A)

        if self.ri_B.dispersive:
            ribc = self.ri_B.complex(wavelength)
        else:
            ribc = complex(self.ri_B)

        ema = getattr(structure, "ema", "linear")
        dp = getattr(structure, "depolarisation_factor", 1 / 3)

        slabs = np.zeros((1, 5))
        N_avg = overall_ri(
            riac, ribc, vf_B=self.vf_B.value, ema=ema, depolarisation_factor=dp
        )
        slabs[0, 0] = self.thick.value
        slabs[0, 1] = np.real(N_avg)
        slabs[0, 2] = np.imag(N_avg)
        slabs[0, 3] = self.rough.value
        return slabs


class StructureSE(Structure):
    """
    Represents the interfacial Structure of an Ellipsometry sample.
    Successive Components are added to the Structure to construct the
    interface.

    Parameters
    ----------
    components : sequence
        A sequence of ComponentSE to initialise the Structure.
    name : str
        Name of this structure
    solvent : ScattererSE
        Specifies the refractive index of the solvent used for solvation. If no
        solvent is specified then the RI of the solvent is assumed to be
        the RI of `Structure[-1].slabs()[-1]` (after any possible slab order
        reversal).
    reverse_structure : bool
        If `StructureSE.reverse_structure` is `True` then the slab
        representation produced by `StructureSE.slabs` is reversed. The sld
        profile and calculated reflectivity will correspond to this
        reversed structure.
    contract : float
        If contract > 0 then an attempt to contract/shrink the slab
        representation is made. Use larger values for coarser
        profiles (and vice versa). A typical starting value to try might
        be 1.0.
    wavelength : float, None
        Wavelength the sample was measured at.
    ema : {'linear', 'maxwell-garnett', 'bruggeman'}
        Specifies the effective medium approximation for how the RI of a
        Component is mixed with the RI of the solvent. Further details
        regarding mixing are explained in the `slabs` method.
    depolarisation_factor : float, int
        The depolarisation factor is used only in the EMA calculations for
        the Maxwell-Garnett and Bruggeman methods. It describes the
        electric field screening: 0 prescribing no screening and 1
        prescribing maximum screening.
    """

    def __init__(
        self,
        components=(),
        name="",
        solvent=None,
        reverse_structure=False,
        contract=0,
        wavelength=None,
        ema="linear",
        depolarisation_factor=1 / 3,
    ):
        super().__init__()
        self._name = name
        self.solvent = solvent
        self._reverse_structure = bool(reverse_structure)

        #: **float** if contract > 0 then an attempt to contract/shrink the
        #: slab representation is made. Use larger values for coarser profiles
        #: (and vice versa). A typical starting value to try might be 1.0.
        self.contract = contract

        # used for energy dispersive measurements.
        self.wavelength = wavelength

        # if you provide a list of components to start with, then initialise
        # the structure from that
        self.data = [c for c in components if isinstance(c, ComponentSE)]

        self.ema = ema
        self._depolarisation_factor = depolarisation_factor

    sld_profile = None

    def __copy__(self):
        s = StructureSE(name=self.name, solvent=self._solvent)
        s.data = self.data.copy()
        return s

    @property
    def solvent(self):
        if self._solvent is None:
            if not self.reverse_structure:
                solv_slab = self[-1].slabs(self)
            else:
                solv_slab = self[0].slabs(self)
            return RI(complex(solv_slab[-1, 1], solv_slab[-1, 2]))
        else:
            return self._solvent

    @solvent.setter
    def solvent(self, ri):
        if ri is None:
            self._solvent = None
        elif isinstance(ri, ScattererSE):
            # don't make a new SLD object, use its reference
            self._solvent = ri
        else:
            solv = RI(ri)
            self._solvent = solv

    @property
    def depolarisation_factor(self):
        """ """
        return self._depolarisation_factor

    @depolarisation_factor.setter
    def depolarisation_factor(self, value):
        if 0 <= float(value) <= 1:
            self._depolarisation_factor = float(value)
        else:
            raise ValueError(
                "Depolarisation factor needs to be a float in [0, 1]."
            )

    def append(self, item):
        """
        Append a :class:`Component` to the Structure.

        Parameters
        ----------
        item: refnx.reflect.Component
            The component to be added.
        """
        if isinstance(item, ScattererSE):
            self.append(item())
            return

        if not isinstance(item, ComponentSE):
            raise ValueError(
                "You can only add ComponentSE objects to a structure"
            )
        super().append(item)

    def slabs(self, **kwds):
        r"""
        The slab representation of this structure.

        Returns
        -------
        slabs : :class:`np.ndarray`
            Slab representation of this structure.
            Has shape (N, 5).

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               *overall* RI.real of layer N (material AND solvent)
            - slab[N, 2]
               *overall* RI.imag of layer N (material AND solvent)
            - slab[N, 3]
               roughness between layer N and N-1
            - slab[N, 4]
               volume fraction of solvent in layer N.

        Notes
        -----
        If `Structure.reversed is True` then the slab representation order is
        reversed. The slab order is reversed before the solvation calculation
        is done. I.e. if `Structure.solvent == 'backing'` and
        `Structure.reversed is True` then the material that solvates the system
        is the component in `Structure[0]`, which corresponds to
        `Structure.slab[-1]`.

        Users can simulate mixing between two adjacent layers by specifying a
        volume fraction of solvent (`vfsolv`). The `overall_ri` function then
        performs the EMA using the specified method: 'linear',
        'maxwell-garnett' or 'bruggeman'. All EMA calculations are performed
        by using the complex dielectric function (i.e., square of refractive
        index and extinction coefficient).
        For a host layer (e_h) with volume fraction (vf) of impurities (e_i),
        the overall RI is calculated by

        >>> StructureSE.ema = 'linear'
        e_linear = e_h * (1 - vf) + e_i * vf

        >>> StructureSE.ema = 'maxwell-garnett'
        >>> StructureSE.depolarisation_factor = 1/3
        top = e_h + (depolarisation_factor * (1 - vf) + vf) * (e_i - e_h)
        bottom = e_h + depolarisation_factor * (1 - vf) * (e_i - e_h)
        e_MG = e_h * top_r / bottom_r

        >>> StructureSE.ema = 'bruggeman'
        >>> StructureSE.depolarisation_factor = 1/3
        b = e_h * ((1 - vf) - depolarisation_factor) + e_i * (vf - depolarisation_factor)
        e_BG = (b + np.sqrt(b**2 - 4 * (depolarisation_factor - 1) *
                                 (vf * e_h * e_i * depolarisation_factor
                                  ))) / (2 * (1 - depolarisation_factor))
        """

        if not len(self):
            return None

        if not (
            isinstance(self.data[-1], (SlabSE))
            and isinstance(self.data[0], (SlabSE))
        ):
            raise ValueError(
                "The first and last Components in a StructureSE need to be SlabsSE"
            )
        # over-ride the wavelength
        if "wavelength" in kwds:
            self.wavelength = float(kwds["wavelength"])

        # Each layer can be given a different type of roughness profile
        # that defines transition between successive layers.
        # The default interface is specified by None (= Gaussian roughness)
        interfaces = flatten(self.interfaces)
        if all([i is None for i in interfaces]):
            # if all the interfaces are Gaussian, then simply concatenate
            # the default slabs property of each component.
            sl = [c.slabs(structure=self) for c in self.components]

            try:
                slabs = np.concatenate(sl)
            except ValueError:
                # some of slabs may be None. np can't concatenate arr and None
                slabs = np.concatenate([s for s in sl if s is not None])
        else:
            # there is a non-default interfacial roughness, create a microslab
            # representation
            slabs = self._micro_slabs()

        # if the slab representation needs to be reversed.
        reverse = self.reverse_structure
        if reverse:
            roughnesses = slabs[1:, 3]
            slabs = np.flipud(slabs)
            slabs[1:, 3] = roughnesses[::-1]
            slabs[0, 3] = 0.0

        if (slabs[:, 4] > 0).any():
            # overall SLD is a weighted average of the vfs and slds
            # accessing self.solvent leads to overhead from object
            # creation.
            if self._solvent is not None:
                solv = self._solvent
            else:
                # we should always choose the solvating material to be the last
                # slab. If the structure is not reversed then you want the last
                # slab. If the structure is reversed then you should want to
                # use the first slab, but the code block above reverses the
                # slab order, so we still want the last one
                solv = complex(slabs[-1, 1], slabs[-1, 2])

            slabs[1:-1] = self.overall_ri(slabs[1:-1], solv)

        if self.contract > 0:
            return contract_by_area(slabs, self.contract)
        else:
            return slabs

    def ri_profile(self, z=None, align=0, max_delta_z=None):
        """
        Calculates an RI profile, as a function of distance through the
        interface.

        Parameters
        ----------
        z : float
            Interfacial distance (Angstrom) measured from interface between the
            fronting medium and the first layer.
        align: int, optional
            Places a specified interface in the slab representation of a
            Structure at z = 0. Python indexing is allowed, e.g. supplying -1
            will place the backing medium at z = 0.
        max_delta_z : {None, float}, optional
            If specified this will control the maximum spacing between SLD
            points. Only used if `z is None`.

        Returns
        -------
        ri : float
            refractive index

        Notes
        -----
        This can be called in vectorised fashion.
        """
        slabs = self.slabs()
        if (
            (slabs is None)
            or (len(slabs) < 2)
            or (not isinstance(self.data[0], SlabSE))
            or (not isinstance(self.data[-1], SlabSE))
        ):
            raise ValueError(
                "Structure requires fronting and backing"
                " Slabs in order to calculate."
            )

        zed, sld = sld_profile(slabs, z=z, max_delta_z=max_delta_z)

        offset = 0
        if align != 0:
            align = int(align)
            if align >= len(slabs) - 1 or align < -1 * len(slabs):
                raise RuntimeError(
                    "abs(align) has to be less than " "len(slabs) - 1"
                )
            # to figure out the offset you need to know the cumulative distance
            # to the interface
            slabs[0, 0] = slabs[-1, 0] = 0.0
            if align >= 0:
                offset = np.sum(slabs[: align + 1, 0])
            else:
                offset = np.sum(slabs[:align, 0])

        return zed - offset, sld

    def plot(self, pvals=None, samples=0, fig=None, align=0):
        """
        Plot the structure.

        Requires matplotlib be installed.

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying
        samples: number
            If this structures constituent parameters have been sampled, how
            many samples you wish to plot on the graph.
        fig: Figure instance, optional
            If `fig` is not supplied then a new figure is created. Otherwise
            the graph is created on the current axes on the supplied figure.
        align: int, optional
            Aligns the plotted structures around a specified interface in the
            slab representation of a Structure. This interface will appear at
            z = 0 in the sld plot. Note that Components can consist of more
            than a single slab, so some thought is required if the interface to
            be aligned around lies in the middle of a Component. Python
            indexing is allowed, e.g. supplying -1 will align at the backing
            medium.

        Returns
        -------
        fig, ax : :class:`matplotlib.Figure`, :class:`matplotlib.Axes`
          `matplotlib` figure and axes objects.

        """
        import matplotlib.pyplot as plt

        params = self.parameters

        if pvals is not None:
            params.pvals = pvals

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        if samples > 0:
            saved_params = np.array(params)
            # Get a number of chains, chosen randomly, and plot the model.
            for pvec in self.parameters.pgen(ngen=samples):
                params.pvals = pvec

                ax.plot(*self.ri_profile(align=align), color="k", alpha=0.01)

            # put back saved_params
            params.pvals = saved_params

        ax.plot(*self.sld_profile(align=align), color="red", zorder=20)
        ax.set_ylabel("RI$")
        ax.set_xlabel("z / $\\AA$")

        return fig, ax

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
        elif isinstance(other, ScattererSE):
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

        Examples
        --------

        >>> air = RI(dispersion=[1,0], name='air')
        >>> sio2 = RI(dispersion=[1.46, 0], name='SiO2')
        >>> si = RI(dispersion=[3.84, 0.016], name='Si')
        >>> structure = StructureSE()
        >>> structure = air() | sio2(20) | si()
        """
        # c = self | other
        p = StructureSE()
        p |= self
        p |= other
        return p

    def overall_ri(self, slabs, solvent):
        """
        Calculates the overall refractive index of the material and solvent RI
        in a layer.

        Parameters
        ----------
        slabs : np.ndarray
            Slab representation of the layers to be averaged.
        solvent : complex or ScattererSE
            RI of solvating material.

        Returns
        -------
        averaged_slabs : np.ndarray
            the averaged slabs.

        """
        solv = solvent
        if isinstance(solvent, ScattererSE):
            solv = solvent.complex(self.wavelength)

        vf = slabs[..., 4]
        N = slabs[..., 1] + slabs[..., 2] * 1j

        N_avg = overall_ri(
            N,
            solv,
            vf_B=vf,
            ema=self.ema,
            depolarisation_factor=self.depolarisation_factor,
        )
        slabs[..., 1] = np.real(N_avg)
        slabs[..., 2] = np.imag(N_avg)

        return slabs


def overall_ri(ri_A, ri_B, vf_B=0.0, ema="linear", depolarisation_factor=1 / 3):
    """
    Calculates the overall refractive index of two materials.

    Parameters
    ----------
    ri_A: complex, array-like
        RI of material A
    ri_B: complex
        RI of material B
    vf_B: float, optional
        volume fraction of material B. The volume fraction of A is calculated
        as ``1 - vf_B``.
    ema : {'linear', 'maxwell-garnett', 'bruggeman'}
        Specifies how refractive indices are mixed together.
    depolarisation_factor : float, optional
        Depolarisation factor. Default is 1/3.

    Returns
    -------
    ri_avg : complex
        the averaged material RI
    """
    # E is the complex dielectric function
    E_a = np.power(ri_A, 2)
    E_b = np.power(ri_B, 2)

    if ema == "linear":
        E_avg = (1 - vf_B) * E_a + vf_B * E_b

    elif ema == "maxwell-garnett":
        top = E_a + (depolarisation_factor * (1 - vf_B) + vf_B) * (E_b - E_a)
        bottom = E_a + depolarisation_factor * (1 - vf_B) * (E_b - E_a)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            v = top / bottom
            v = np.where(np.isfinite(v), v, 0)
            E_avg = E_a * v

    elif ema == "bruggeman":
        # The solution to the Bruggeman EMA method is solved using the
        # quadratic equation, only one of which is physically reasonable.

        b = E_a * ((1 - vf_B) - depolarisation_factor) + E_b * (
            vf_B - depolarisation_factor
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            v = (
                b
                + np.sqrt(
                    b**2
                    - 4
                    * (depolarisation_factor - 1)
                    * (E_a * E_b * depolarisation_factor)
                )
            ) / (2 * (1 - depolarisation_factor))
            E_avg = np.where(np.isfinite(v), v, 0)
    else:
        raise RuntimeError("No other method of mixing is known")

    ri_avg = np.sqrt(E_avg)
    return ri_avg
