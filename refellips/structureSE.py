"""
Created on Mon Sep 21 11:10:38 2020.

@author: Isaac
"""
from refnx.reflect.structure import Scatterer
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
import numpy as np
import os
import warnings

class RI(Scatterer):
    """
    Object representing a materials wavelength-dependent refractive index.

    A concern is how it needs to be linked to a model. This is to get around
    a major rewrite of refnx, but isn't the most elegant system.

    Another issue is that optical parameters are supplied in units of micro
    meters ('cause thats what seems to be used in refractive index repos and
    cauchy models), the wavelength of the incident radiation is supplied in
    nanometers (thats typical) and the fitting is done in angstroms. Very
    unpleasent.

    Parameters
    ----------
    value : tuple, string
        Scattering length density of a material.
        Units (10**-6 Angstrom**-2)
    A : float or parameter
        Cauchy parameter A. If not none RI will use the cauchy model.
        Default None.
    B : float or parameter
        Cauchy parameter B in um^2. Default 0.
    C : float or parameter
        Cauchy parameter C in um^4. Default 0.
    name : str, optional
        Name of material.

    Notes
    -----
    An SLD object can be used to create a Slab:

    >>> # an SLD object representing Silicon Dioxide
    >>> sio2 = SLD(3.47, name='SiO2')
    >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
    >>> sio2_layer = sio2(20, 3)

    The SLD object can also be made from a complex number, or from Parameters

    >>> sio2 = SLD(3.47+0.01j)
    >>> re = Parameter(3.47)
    >>> im = Parameter(0.01)
    >>> sio2 = SLD(re)
    >>> sio2 = SLD([re, im])
    """

    def __init__(self, value=None, A=None, B=0, C=0, name=""):
        if type(value) is str and name == "": # if there is no name get it from the path
            name = os.path.basename(value).split('.')[0]

        super(RI, self).__init__(name=name)
        
        assert np.logical_xor(value is None, A is None),\
        'Supply either values or cauchy parameters'

        if value is not None:
            if type(value) is str:

                try:
                    self._wav, self._RI, self._EC = np.loadtxt(value, skiprows=1,
                                                               delimiter=',', encoding='utf8').T
                except ValueError:
                    self._wav, self._RI = np.loadtxt(value, skiprows=1,
                                                     delimiter=',',
                                                     usecols=[0, 1], encoding='utf8').T
                    self._EC = np.zeros_like(self._wav)

            elif len(value) == 2:
                self._RI, self._EC = value
                self._wav = None
            elif len(value) == 3:
                self._wav, self._RI, self._EC = value
            else:
                raise TypeError ('format not recognised')
            # convert wavelength from um to nm
            self._wav = self._wav * 1000
        else:
            self._wav = None
            self._RI = None
            self._EC = None



        self.model = None
        self.set_wav = None
        self._default_wav = 658
        self._parameters = Parameters(name=name)

        if A is not None:
            self.A = possibly_create_parameter(A, name=f'{name} - cauchy A')
            self.B = possibly_create_parameter(B, name=f'{name} - cauchy B')
            self.C = possibly_create_parameter(C, name=f'{name} - cauchy C')
            self._parameters.extend([self.A, self.B, self.C])

        # The RI needs access to the model to calculate the refractive index.
        # Can't think of a better way of doing this
        # reflect_modelSE is going to auto-link this when its called.

    @property
    def real(self):
        """Refractive index, n."""

        if self.model is not None:
            wavelength = self.model.wav
        elif self.set_wav is not None:
            wavelength = self.set_wav
        else:
            wavelength = self._default_wav
            warnings.warn('Using default wavelength (model not linked)')

        if np.any(self._wav):
            # TODO - raise a warning if the wavelength supplied is outside the
            # wavelength range covered by the data file.

            return Parameter(np.interp(wavelength,
                                       self._wav, self._RI))

        elif self.A is not None:
            return Parameter(self.A.value + (self.B.value*1000**2)/(wavelength**2)\
                                          + (self.C.value**1000**4)/(wavelength**4))
        else:
            return Parameter(value=self._RI)

    @property
    def imag(self, wavelength=None):
        """Extinction coefficent, k."""
        
        if self.model is not None:
            wavelength = self.model.wav
        elif self.set_wav is not None:
            wavelength = self.set_wav
        else:
            wavelength = self._default_wav
            warnings.warn('Using default wavelength (model not linked)')

        if np.any(self._wav):
            # TODO - raise a warning if the wavelength supplied is outside the
            # wavelength range covered by the data file.

            return Parameter(np.interp(wavelength,
                                       self._wav, self._EC))
        elif self.A is not None:
            return Parameter(0)
        else:
            return Parameter(value=self._EC)

    @property
    def parameters(self):
        return self._parameters

    def __repr__(self):
        return str(f'n: {self.real.value}, k: {self.imag.value}')

    def __complex__(self):
        sldc = complex(self.real.value, self.imag.value)
        return sldc
