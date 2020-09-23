# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:05:38 2020

@author: Isaac
"""
import numpy as np
from tmm import coh_tmm


from refnx.analysis import (
    Parameters,
    Parameter,
    possibly_create_parameter,
    Transform,
)

class ReflectModelSE(object):
    r"""
    Parameters
    ----------
    structure : refnx.reflect.Structure
        The interfacial structure.
    name : str, optional
        Name of the Model
    """

    def __init__(
        self,
        structure,
        wavelength,
        delta_offset = 0,
        name=None,
    ):
        self.name = name
        self.DeltaOffset = delta_offset
        self._parameters = None

        # to make it more like a refnx.analysis.Model
        self.fitfunc = None

        # all reflectometry models need a scale factor and background
        self._wav = possibly_create_parameter(wavelength, name="wavelength")

        self._structure = None
        self.structure = structure

        self.DeltaOffset = possibly_create_parameter(delta_offset, name='delta offset')

        # THIS IS REALLY QUENSTIONABLE
        for x in self._structure:
            try:
                x.sld.model = self
            except AttributeError:
                print ("it appears you are using SLD's instead of RIs")

    def __call__(self, aoi, p=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity
        """
        return self.model(aoi, p=p)

    def __repr__(self):
        return (
            f"ReflectModel({self._structure!r}, name={self.name!r},"
            f" wavelength={self.wav!r},"
            f" delta_offset = {self.delOffset!r} "
        )

    @property
    def wav(self):
        r"""
        :class:`refnx.analysis.Parameter` - all model values are multiplied by
        this value before the background is added.

        """
        return self._wav

    @wav.setter
    def wav(self, value):
        self._wav.value = value

    @property
    def delOffset(self):
        """
        :class:`refnx.analysis.Parameter` - the calculated delta offset specific 
        to the ellipsometer and experimental setup used.

        """
        return self.DeltaOffset

    @delOffset.setter
    def delOffset(self, value):
        self.DeltaOffset.value = value


    def model(self, aoi, p=None):
        r"""
        Calculate the reflectivity of this model

        Parameters
        ----------
        aoi : float or np.ndarray
            aoi values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        """
        if p is not None:
            self.parameters.pvals = np.array(p)

        return Delta_Psi_TMM(
            AOI=aoi,
            layers=self.structure.slabs()[..., :4],
            wavelength=self.wav.value,
            delta_offset=self.delOffset.value
        )

    def logp(self):
        r"""
        Additional log-probability terms for the reflectivity model. Do not
        include log-probability terms for model parameters, these are
        automatically included elsewhere.

        Returns
        -------
        logp : float
            log-probability of structure.

        """
        return self.structure.logp()

    @property
    def structure(self):
        r"""
        :class:`refnx.reflect.Structure` - object describing the interface of
        a reflectometry sample.

        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        p = Parameters(name="instrument parameters")
        p.extend([self.wav, self.delOffset])

        self._parameters = Parameters(name=self.name)
        self._parameters.extend([p, structure.parameters])

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters` - parameters associated with this
        model.

        """
        self.structure = self._structure
        return self._parameters
    
    
def Delta_Psi_TMM(AOI, layers, wavelength, delta_offset):
    """
    Get delta and psi using the transfer matrix method.
    
    This is a copy of the refnx Abeles code. If we can wrap this around tmm
    we should be able to get it into refnx propper.
    
    There will be some work to smooth things out upstream (SLD objects etc.)
    but that should become clearer when this is written.
    
    Lets do one function call for each wavelength for now. To start with its
    probably easiest just to impliment it for one wavelength
    
    Parameters
    ----------
    AOI: array_like
        the angle of incidence values required for the calculation.
        Units = degrees
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - refractive index of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - extinction coefficent of fronting (/1e-6 Angstrom**-2)
        layers[N, 0] - thickness of layer N
        layers[N, 1] - refractive index of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - extinction coefficent of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - roughness between layer N-1/N
        layers[-1, 1] - refractive index of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - extinction coefficent of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - roughness between backing and last layer

    Returns
    -------
    Psi: np.ndarray
        Calculated Psi values for each aoi value.
    Delta: np.ndarray
        Calculated Delta values for each aoi value.


    """
    AOI = np.array(AOI)
    AOI = AOI*(np.pi/180)

    RIs        = layers[:, 1] + layers[:, 2]*1j
    thicks     = layers[:, 0]/10 #Ang to nm
    thicks[0]  = np.inf
    thicks[-1] = np.inf
    
    psi   = np.zeros_like(AOI)
    delta = np.zeros_like(AOI)
    

    for idx, aoi in enumerate(AOI):
        s_data = coh_tmm('s', n_list=RIs, d_list=thicks, th_0=aoi, lam_vac=wavelength)
        p_data = coh_tmm('p', n_list=RIs, d_list=thicks, th_0=aoi, lam_vac=wavelength)
        rs = s_data['r']
        rp = p_data['r']
    
        psi[idx]    = np.arctan(abs(rp/rs))
        delta[idx]  = np.angle(1/(-rp/rs))+np.pi
    return psi*(180/np.pi), delta*(180/np.pi)+delta_offset