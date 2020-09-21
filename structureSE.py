"""
Created on Mon Sep 21 11:10:38 2020.

@author: Isaac
"""
from refnx.reflect.structure import Scatterer
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
import numpy as np


class RI(Scatterer):
    """
    Object representing a materials wavelength-dependent refractive index.

    Currently is the minimum viable solution - does not deal with cauchy model.

    A concern is how it needs to be linked to a model. This is to get around
    a major rewrite of refnx, but isn't the most elegant system.

    Parameters
    ----------
    value : tuple, string
        Scattering length density of a material.
        Units (10**-6 Angstrom**-2)
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

    def __init__(self, value, name=""):
        super(RI, self).__init__(name=name)

        # TODO: Deal with Cauchy parameters
        if type(value) is str:
            try:
                self._wav, self._RI, self._EC = np.loadtxt(value, skiprows=1,
                                                           delimiter=',').T
            except ValueError:
                self._wav, self._RI = np.loadtxt(value, skiprows=1,
                                                 delimiter=',',
                                                 usecols=[0, 1]).T
                self._EC = np.zeros_like(self._wav)
        elif len(value) == 2:
            self._RI, self._EC = value
            self._wav = None
        elif len(value) == 3:
            self._wav, self._RI, self._EC = value
        else:
            raise Exception('Format is not understand')

        self._wav = self._wav * 1000  # convert wavelength from um to nm

        # The RI needs access to the model to calculate the refractive index.
        # Can't think of a better way of doing this
        # reflect_modelSE is going to auto-link this when its called.

        self.model = None
        self._parameters = Parameters(name=name)
#         self._parameters.extend(Cauchy Parameters])

    @property
    def real(self):
        """Refractive index, n."""
        if np.any(self._wav):
            if self.model:
                return Parameter(np.interp(self.model.wav,
                                           self._wav, self._RI))
            else:
                raise Exception('you need to link the model\
                                 to supply a wavelength.')
        else:
            return Parameter(value=self._RI)

    @property
    def imag(self):
        """Extinction coefficent, k."""
        if np.any(self._wav):
            if self.model:
                return Parameter(np.interp(self.model.wav,
                                           self._wav, self._EC))
            else:
                raise Exception('you need to link the model\
                                 to supply a wavelength.')
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
