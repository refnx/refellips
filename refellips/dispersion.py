import numpy as np
import os
import os.path
import glob
from pathlib import PurePath
import scipy.fftpack as ft


from refnx.analysis import (
    Parameters,
    possibly_create_parameter,
    sequence_to_parameters,
)

from .structureSE import ScattererSE, nm_eV_conversion


# list of material dispersion curves distributed with refellips
_pth = os.path.dirname(os.path.abspath(__file__))
_material_files = glob.glob(os.path.join(_pth, "materials/*.csv"))
materials = [os.path.basename(m)[:-4] for m in _material_files]


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
    dispersion : {str, Path, tuple, np.ndarray)
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
            if type(dispersion) is str or isinstance(dispersion, PurePath):
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


class Sellmeier(ScattererSE):
    r"""
    Dispersion curves for Sellmeier oscillators.

    Parameters
    ----------
    Am: {float, Parameter}
        Amplitude of Sellmeier in μm.
    En: {float, Parameter}
        Center energy of oscillator in μm.
    P: {float, Parameter}
        Position of a pole in μm.
    Einf: {float, Parameter}
        Offset term
    wavelength : float
        default wavelength for calculation (nm)
    name : str, optional
        Name of material.

    Notes
    -----
    Calculates dispersion curves of a Sellmeier oscillator as implemented in
    CompleteEase.
    CompleteEase Manual, Chapter 9, pg 9-306, J.A. Woollam Co., 2014.

    .. math::

    n = \sqrt{ \varepsilon (\infty) + \frac{Am \lambda^2}{\lambda^2 - En^2} - P\lambda^2}

    Examples
    --------
    >>> # Create a Sellmeier oscillator
    >>> sell = Sellmeier(2, 0.1, 0.11, Einf=1)
    >>> sell.complex(658)  # calculates the refractive index at 658 nm.
    """

    def __init__(self, Am, En, P, Einf=1, wavelength=658, name=""):
        super().__init__(name=name, wavelength=wavelength)

        self.Am = possibly_create_parameter(Am, name=f"{name} - sellmeier Am")
        self.En = possibly_create_parameter(En, name=f"{name} - sellmeier En")
        self.P = possibly_create_parameter(P, name=f"{name} - sellmeier P")
        self.Einf = possibly_create_parameter(
            Einf, name=f"{name} - sellmeier Einf"
        )

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.Am, self.En, self.P, self.Einf])

    @property
    def parameters(self):
        return self._parameters

    def complex(self, wavelength):
        """
        Calculate a complex RI for the given Sellmeier oscillator

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

        # Convert between μm & nm (constants are typically given in μm)
        wav *= 1e-3

        real = np.sqrt(
            self.Einf.value
            + (self.Am.value * wav**2) / (wav**2 - self.En.value**2)
            - (self.P.value * wav**2)
        )
        return real + 1j * 0.0

    def epsilon(self, wavelength):
        """
        The complex dielectric function for the oscillator
        """
        wav = self.wavelength
        if np.any(wavelength):
            wav = wavelength

        # Convert between μm & nm (constants are typically given in μm)
        wav *= 1e-3

        real = (
            self.Einf.value
            + (self.Am.value * wav**2) / (wav**2 - self.En.value**2)
            - (self.P.value * wav**2)
        )

        return real + 1j * 0


class Lorentz(ScattererSE):
    r"""
    Dispersion curves for Lorentz oscillators.

    Parameters
    ----------
    Am: {float, Parameter, sequence}
        Amplitude of Lorentzian
    Br: {float, Parameter, sequence}
        Broadening of oscillator
    En: {float, Parameter, sequence}
        Centre energy of oscillator (eV)
    Einf: {float, Parameter}
        Offset term
    wavelength : float
        default wavelength for calculation (nm)
    name : str, optional
        Name of material.

    Notes
    -----
    Calculates dispersion curves for *k* oscillators, as implemented in WVASE.
    The model is Kramers-Kronig consistent.
    The parameters for constructing this object should have
    `len(Am) == len(Br) == len(En) == k`, or be single float/Parameter.

    ..math::

    \tilde{\varepsilon}(h\nu)=\varepsilon_{1\infty }+\sum_{k}\frac{A_{k}}{E_{k}^2 - (h\nu)^2-iB_kh\nu}

    Examples
    --------
    >>> # Create a single Lorentz oscillator
    >>> Lorentz(5, 0.25, 2, Einf=1)
    >>> # Create a 2 oscillator dispersion curve
    >>> lo = Lorentz([5, 10], [0.25, 0.5], [2, 4], Einf=2)
    >>> lo.complex(658)  # calculates the refractive index at 658 nm.
    """

    def __init__(self, Am, Br, En, Einf=1, wavelength=658, name=""):
        super().__init__(name=name, wavelength=wavelength)

        self._parameters = Parameters(name=name)
        self.Am = sequence_to_parameters([Am])
        self.Br = sequence_to_parameters([Br])
        self.En = sequence_to_parameters([En])
        if not (len(self.Am) == len(self.Br) == len(self.En)):
            raise ValueError("A, B, E all have to be the same length")

        self._parameters.extend([self.Am, self.Br, self.En])
        self.Einf = possibly_create_parameter(Einf)
        self._parameters.append(self.Einf)

    @property
    def parameters(self):
        return self._parameters

    def epsilon(self, energy):
        """
        The complex dielectric function for the oscillator
        """
        A = np.array(self.Am)[:, None]
        B = np.array(self.Br)[:, None]
        E = np.array(self.En)[:, None]
        _e = np.asfarray(energy)
        v = A / (E**2 - _e**2 - 1j * B * _e)
        r = np.atleast_1d(np.sum(v, axis=0) + self.Einf.value)

        if np.isscalar(energy) and len(r) == 1:
            return r[0]

        return r


class Gauss(ScattererSE):
    """
    Dispersion curves for Gaussian oscillators.

    Parameters
    ----------
    Am: {float, Parameter, sequence}
        Amplitude of Gaussian
    Br: {float, Parameter, sequence}
        Broadening of oscillator
    En: {float, Parameter, sequence}
        Centre energy of oscillator (eV)
    Einf: {float, Parameter}
        Offset term
    wavelength : float
        default wavelength for calculation (nm)
    name : str, optional
        Name of material.

    Notes
    -----
    Calculates dispersion curves for *k* Gaussian oscillators.
    The model is Kramers-Kronig consistent.
    The parameters for constructing this object should have
    `len(Am) == len(Br) == len(En) == k`, or be single float/Parameter.
    """

    def __init__(self, Am, Br, En, Einf=1, wavelength=658, name=""):
        super().__init__(name=name, wavelength=wavelength)

        self._parameters = Parameters(name=name)
        self.Am = sequence_to_parameters([Am])
        self.Br = sequence_to_parameters([Br])
        self.En = sequence_to_parameters([En])
        if not (len(self.Am) == len(self.Br) == len(self.En)):
            raise ValueError("A, B, E all have to be the same length")

        self.Einf = possibly_create_parameter(Einf)
        self._parameters.extend([self.Am, self.Br, self.En])
        self._parameters.append(self.Einf)

    @property
    def parameters(self):
        return self._parameters

    def epsilon(self, energy):
        """
        The complex dielectric function for the oscillator
        """
        A = np.array(self.Am)[:, None]
        B = np.array(self.Br)[:, None]
        E = np.array(self.En)[:, None]
        energies = np.asfarray(energy)

        # TODO cache if params don't change
        _e_pad = np.linspace(-20, 20, 2048)
        sigma = B / 2 / np.sqrt(np.log(2))
        e2 = A * np.exp(-(((_e_pad - E) / sigma) ** 2))
        e2 -= A * np.exp(-(((_e_pad + E) / sigma) ** 2))

        # e1 is Kramers-Kronig consistent via Hilbert transform
        # e1 = ft.hilbert(e2) + self.Einf.value
        e1 = np.array([ft.hilbert(_e2) for _e2 in e2])

        e1 = np.sum(e1, axis=0) + self.Einf.value
        e2 = np.sum(e2, axis=0)

        # (linearly) interpolate to find epsilon at given energy
        _e1 = np.interp(energies, _e_pad, e1)
        _e2 = np.interp(energies, _e_pad, e2)
        r = np.atleast_1d(_e1 + 1j * _e2)
        if np.isscalar(energy) and len(r) == 1:
            return r[0]
        return r


class TaucLorentz(ScattererSE):
    """
    Dispersion curves for Tauc-Lorentz oscillators. The model works well for
    amorphous materials in the visible range.

    Parameters
    ----------
    Am: {float, Parameter, sequence}
        Amplitude of absorption. Typically in [10, 200]
    C: {float, Parameter, sequence}
        Lorentz broadening of oscillator (eV). Typically in [0, 10].
        ``C`` should be less than ``2 * En``
    En: {float, Parameter, sequence}
        Lorentz resonance energy (eV). ``En`` should be greater than ``Eg``.
    Eg: {float, Parameter}
        Common bandgap energy (eV) for all oscillators
    Einf: {float, Parameter}
        Offset term
    wavelength : float
        default wavelength for calculation (nm)
    name : str, optional
        Name of material

    Notes
    -----
    Calculates dispersion curves for *k* Tauc-Lorentz oscillators.
    The model is Kramers-Kronig consistent.
    The parameters for constructing this object should have
    `len(Am) == len(C) == len(En) == k`, or be single float/Parameter.

    Implemented using the equations from
    `Horiba technical note <https://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Tauc-Lorentz_Dispersion_Formula.pdf>`_
    and also the WVASE manual and https://en.wikipedia.org/wiki/Tauc%E2%80%93Lorentz_model.

    The Horiba technical note gives parameters for many materials.

    * G.E. Jellision and F.A. Modine, Appl. Phys. Lett. 69 (3), 371-374 (1996)
    * Erratum, G.E. Jellison and F.A. Modine, Appl. Phys. Lett 69 (14), 2137 (1996)
    * H. Chen, W.Z. Shen, Eur. Phys. J. B. 43, 503-507 (2005)
    """

    def __init__(self, Am, C, En, Eg, Einf=1, wavelength=658, name=""):
        super().__init__(name=name, wavelength=wavelength)

        self._parameters = Parameters(name=name)
        self.Am = sequence_to_parameters([Am])
        self.C = sequence_to_parameters([C])
        self.En = sequence_to_parameters([En])
        if not (len(self.Am) == len(self.C) == len(self.En)):
            raise ValueError("A, B, E all have to be the same length")

        self.Einf = possibly_create_parameter(Einf)
        self.Eg = possibly_create_parameter(Eg)
        self._parameters.extend([self.Am, self.C, self.En, self.Einf, self.Eg])

    @property
    def parameters(self):
        return self._parameters

    def epsilon(self, energy):
        """
        The complex dielectric function for the oscillator
        """
        A = np.array(self.Am)[:, None]
        C = np.array(self.C)[:, None]
        Ei = np.array(self.En)[:, None]
        Eg = self.Eg.value
        energies = np.asfarray(energy)

        a_ln = (
            (Eg**2 - Ei**2) * energies**2
            + Eg**2 * C**2
            - Ei**2 * (Ei**2 + 3 * Eg**2)
        )
        a_atan = (energies**2 - Ei**2) * (
            Ei**2 + Eg**2
        ) + Eg**2 * C**2
        alpha = np.sqrt(4 * Ei**2 - C**2)
        gamma = np.sqrt(Ei**2 - C**2 / 2)
        zeta4 = (energies**2 - gamma**2) ** 2 + 0.25 * alpha**2 * C**2

        e1 = (
            A
            * C
            * a_ln
            / 2
            / np.pi
            / zeta4
            / alpha
            / Ei
            * np.log(
                (Ei**2 + Eg**2 + alpha * Eg)
                / (Ei**2 + Eg**2 - alpha * Eg)
            )
        )
        e1 -= (
            A
            / np.pi
            * a_atan
            / zeta4
            / Ei
            * (
                np.pi
                - np.arctan((2 * Eg + alpha) / C)
                + np.arctan((alpha - 2 * Eg) / C)
            )
        )
        e1 += (
            2
            * A
            * Ei
            * Eg
            * (energies**2 - gamma**2)
            / np.pi
            / zeta4
            / alpha
            * (np.pi + 2 * np.arctan(2 * (gamma**2 - Eg**2) / alpha / C))
        )
        e1 -= (
            A
            * Ei
            * C
            * (energies**2 + Eg**2)
            / np.pi
            / zeta4
            / energies
            * np.log(np.abs(energies - Eg) / (energies + Eg))
        )
        e1 += (
            2
            * A
            * Ei
            * Eg
            * C
            / np.pi
            / zeta4
            * np.log(
                np.abs(energies - Eg)
                * (energies + Eg)
                / np.sqrt((Ei**2 - Eg**2) ** 2 + Eg**2 * C**2)
            )
        )
        e1 = np.sum(e1, axis=0) + self.Einf.value

        # I don't think the Hilbert Transform works all that well on this
        # dielectric function because the tail drops off v slowly.
        e2 = A * Ei * C * (energies - Eg) ** 2
        e2 /= energies * ((energies**2 - Ei**2) ** 2 + (C * energies) ** 2)
        e2 *= np.heaviside(energies - Eg, 0)
        e2 = np.sum(e2, axis=0)

        r = np.atleast_1d(e1 + 1j * e2)
        if np.isscalar(energy) and len(r) == 1:
            return r[0]
        return r
