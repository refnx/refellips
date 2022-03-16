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


class ScattererSE(Scatterer):
    """
    Abstract base class for something that will have a refractive index.
    Inherited from refnx.reflect.structure.Scatterer
    """

    def __init__(self, name=""):
        self.name = name
        # by default energy dispersive scatterers for ellipsometry are energy dispersive
        self.dispersive = True

    def __str__(self):
        ri = complex(self)
        return "RI = {0}".format(ri)

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
    ema : {'linear', 'maxwell-garnett'}
        Specifies the effective medium approximation for how the RI of a
        Component is mixed with the RI of the solvent.
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
            Scattering length density / 1e-6 Angstrom**-2

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

        return overall_RI(slabs, solv, ema=self.ema)


def overall_RI(slabs, solvent, ema="linear"):
    """
    Calculates the overall refractive index of the material and solvent RI
    in a layer.

    Parameters
    ----------
    slabs : np.ndarray
        Slab representation of the layers to be averaged.
    solvent : complex or RI
        RI of solvating material.
    ema : {'linear', 'maxwell-garnett'}
        Specifies how refractive indices are mixed together

    Returns
    -------
    averaged_slabs : np.ndarray
        the averaged slabs.
    """
    vf = slabs[..., 4]

    if ema == "linear":
        slabs[..., 1:3] = slabs[..., 1:3] ** 2
        slabs[..., 1:3] *= (1 - vf)[..., np.newaxis]

        slabs[..., 1] += solvent.real**2 * vf
        slabs[..., 2] += solvent.imag**2 * vf
    elif ema_method == "maxwell-garnett":
        slabs[..., 1:3] = slabs[..., 1:3] ** 2

        top_r = 2 * (1 - vf) * slabs[..., 1] + (1 + 2 * vf) * solvent.real**2
        bottom_r = (2 + vf) * slabs[..., 1] + (1 - vf) * solvent.real**2
        slabs[..., 1] *= top_r / bottom_r

        top_r = 2 * (1 - vf) * slabs[..., 2] + (1 + 2 * vf) * solvent.imag**2
        bottom_r = (2 + vf) * slabs[..., 2] + (1 - vf) * solvent.imag**2
        slabs[..., 2] *= top_r / bottom_r
    else:
        raise RuntimeError("No other method of mixing is known")

    slabs[..., 1:3] = np.sqrt(slabs[..., 1:3])

    return slabs
