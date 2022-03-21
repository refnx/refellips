""""""
"""
Calculates the ellipsometry parameters Δ and Ψ from a stratified series of
layers.

TBSD 3-Clause License

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


Furthermore, here we adapt code from Byrnes (i.e. the coh_tmm function) under
the MIT license listed below:

Copyright (C) 2012-2020 Steven Byrnes

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# -*- coding: utf-8 -*-

import numpy as np
from numpy.lib.scimath import arcsin
from refellips import StructureSE

EPSILON = np.finfo(np.float64).eps

from refnx.analysis import (
    Parameters,
    Parameter,
    possibly_create_parameter,
    Transform,
)


def interface_r_s(n_i, n_f, th_i, th_f):
    return (n_i * np.cos(th_i) - n_f * np.cos(th_f)) / (
        n_i * np.cos(th_i) + n_f * np.cos(th_f)
    )


def interface_r_p(n_i, n_f, th_i, th_f):
    return (n_f * np.cos(th_i) - n_i * np.cos(th_f)) / (
        n_f * np.cos(th_i) + n_i * np.cos(th_f)
    )


def interface_t_s(n_i, n_f, th_i, th_f):
    return 2 * n_i * np.cos(th_i) / (n_i * np.cos(th_i) + n_f * np.cos(th_f))


def interface_t_p(n_i, n_f, th_i, th_f):
    return 2 * n_i * np.cos(th_i) / (n_f * np.cos(th_i) + n_i * np.cos(th_f))


def coh_tmm(n_list, d_list, th_0, lam_vac):
    """
    Code adapted by that of Byrnes - see https://arxiv.org/abs/1603.02720

    n_list is the list of refractive indices, in the order that the light would
    pass through them. The 0'th element of the list should be the semi-infinite
    medium from which the light enters, the last element should be the semi-
    infinite medium to which the light exits (if any exits).

    th_0 is the angle of incidence: 0 for normal, pi/2 for glancing.
    Remember, for a dissipative incoming medium (n_list[0] is not real), th_0
    should be complex so that n0 sin(th0) is real (intensity is constant as
    a function of lateral position).

    d_list is the list of layer thicknesses (front to back). Should correspond
    one-to-one with elements of n_list. First and last elements should be "inf".

    lam_vac is vacuum wavelength of the light.

    """
    # Convert lists to numpy arrays if they're not already.
    n_list = np.asarray(n_list)
    d_list = np.asfarray(d_list)
    num_layers = n_list.size

    th_0, lam_vac = [np.array(a) for a in np.broadcast_arrays(th_0, lam_vac)]
    orig_shp = th_0.shape
    th_0 = np.ravel(th_0)
    lam_vac = np.ravel(lam_vac)

    # Input tests
    assert (
        np.abs(np.imag(n_list[0] * np.sin(th_0))) < 100 * EPSILON
    ).all(), "Error in n0 or th0!"

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may
    # be complex!
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)

    # (NLAYERS, NUMPNTS)
    th_list = arcsin(n_list[0] * np.sin(th_0[None, :]) / n_list[:, None])

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    # kz_list.shape = (NLAYERS, NUMPOINTS)
    kz_list = 2 * np.pi * n_list[:, None] * np.cos(th_list) / lam_vac

    if num_layers > 2:
        # delta is the total phase accrued by traveling through a given layer.
        # don't work it out for the fronting/backing media
        # delta.shape = (NLAYERS - 2, NUMPNTS)
        delta = kz_list[1:-1] * d_list[1:-1, None]

        # For a very opaque layer, reset delta to avoid divide-by-0 and similar
        # errors. The criterion imag(delta) > 35 corresponds to single-pass
        # transmission < 1e-30 --- small enough that the exact value doesn't
        # matter.
        if (np.imag(delta) > 35).any():
            np.clip(delta.imag, -np.inf, 35, out=delta.imag)

    results = {}

    polarisations = [
        ["s", interface_r_s, interface_t_s],
        ["p", interface_r_p, interface_t_p],
    ]

    for pol, interface_r, interface_t in polarisations:
        # t_list and r_list are transmission and reflection amplitudes,
        # respectively, coming from i, going to j.
        # r_list.shape = (NLAYERS - 1, NUMPOINTS)
        r_list = interface_r(
            n_list[:-1, None], n_list[1:, None], th_list[:-1], th_list[1:]
        )
        t_list = interface_t(
            n_list[:-1, None], n_list[1:, None], th_list[:-1], th_list[1:]
        )

        # At the interface between the (n-1)st and nth material, let v_n be the
        # amplitude of the wave on the nth side heading forwards (away from the
        # boundary), and let w_n be the amplitude on the nth side heading backwards
        # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
        # M_list[n]. M_0 and M_{num_layers-1} are not defined.
        # My M is a bit different than Sernelius's, but Mtilde is the same.

        if num_layers > 2:
            # calculate the characteristic matrices for all the layers
            beta = np.exp(1j * delta)
            beta_inv = 1 / beta

            # M_list00.shape = (NUM_LAYERS - 2, NUMPOINTS)
            M_list00 = beta_inv / t_list[1:]
            M_list01 = r_list[1:] * beta_inv / t_list[1:]
            M_list10 = r_list[1:] * beta / t_list[1:]
            M_list11 = beta / t_list[1:]

        # initial interface
        # Mtilde00.shape = (NUMPOINTS,)
        Mtilde00 = 1 / t_list[0]
        Mtilde01 = r_list[0] / t_list[0]
        Mtilde10 = Mtilde01
        Mtilde11 = Mtilde00

        # propagate characteristic matrices
        for i in range(0, num_layers - 2):
            # matrix multiply Mtilde by characteristic matrix
            p00 = Mtilde00 * M_list00[i, :] + Mtilde01 * M_list10[i, :]
            p01 = Mtilde00 * M_list01[i, :] + Mtilde01 * M_list11[i, :]
            p10 = Mtilde10 * M_list00[i, :] + Mtilde11 * M_list10[i, :]
            p11 = Mtilde10 * M_list01[i, :] + Mtilde11 * M_list11[i, :]

            Mtilde00 = p00
            Mtilde01 = p01
            Mtilde10 = p10
            Mtilde11 = p11

        # Net complex transmission and reflection amplitudes
        r = Mtilde10 / Mtilde00
        t = 1 / Mtilde00
        results[f"r_{pol}"] = r
        results[f"t_{pol}"] = t

    return results


def Delta_Psi_TMM(AOI, layers, wavelength, delta_offset, reflect_delta=False):
    """
    Get delta and psi using the transfer matrix method.

    Parameters
    ----------
    AOI: array_like
        the angle of incidence values required for the calculation.
        Units = degrees
    Wavelength: float
        Wavelength of light. Units = nm
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - refractive index of fronting
        layers[0, 2] - extinction coefficent of fronting
        layers[N, 0] - thickness of layer N
        layers[N, 1] - refractive index of layer N
        layers[N, 2] - extinction coefficent of layer N
        layers[N, 3] - roughness between layer N-1/N (IGNORED!)
        layers[-1, 1] - refractive index of backing
        layers[-1, 2] - extinction coefficent of backing
        layers[-1, 3] - roughness between backing and last layer (IGNORED!)

    Returns
    -------
    Psi: np.ndarray
        Calculated Psi values for each aoi value.
    Delta: np.ndarray
        Calculated Delta values for each aoi value.


    """
    AOI = np.asfarray(AOI)
    AOI = AOI * (np.pi / 180)

    layers[0, 2] = 0  # infinate medium cannot have an extinction coeff
    RIs = layers[:, 1] + layers[:, 2] * 1j
    thicks = layers[:, 0] / 10  # Ang to nm
    thicks[0] = np.inf
    thicks[-1] = np.inf

    results = coh_tmm(n_list=RIs, d_list=thicks, th_0=AOI, lam_vac=wavelength)

    rs = results["r_s"]
    rp = results["r_p"]

    psi = np.arctan(abs(rp / rs))
    delta = np.angle(1 / (-rp / rs)) + np.pi

    if reflect_delta:
        # Different ellipsometers / modelling software have different
        # conventions for what to do with Delta's above 180. WVASE appears to
        # reflect Delta around 180, which is what we have attempted to replicate.
        delta[delta > np.pi] = 2 * np.pi - delta[delta > np.pi]

    return psi * (180 / np.pi), delta * (180 / np.pi) + delta_offset


class ReflectModelSE:
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
        delta_offset=0,
        name=None,
    ):
        if not isinstance(structure, StructureSE):
            raise ValueError("structure must be a StructureSE instance")

        self.name = name
        self._parameters = None
        self._flip_delta = False

        self.delta_offset = possibly_create_parameter(
            delta_offset, name="delta offset"
        )

        # to make it more like a refnx.analysis.Model
        self.fitfunc = None

        self._structure = None
        self.structure = structure

    def __call__(self, wavelength_aoi, p=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        wavelength_aoi : array-like
            An array of shape (N, 2) corresponding to the wavelengths (nm) and
            angle of incidences (deg) the ellipsometric measurements were
            performed at.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model

        Returns
        -------
        psi, delta : np.ndarray
        """
        return self.model(wavelength_aoi, p=p)

    def __repr__(self):
        return (
            f"ReflectModel({self._structure!r}, name={self.name!r},"
            f" delta_offset = {self.delta_offset.value!r} "
        )

    def model(self, wavelength_aoi, p=None):
        r"""
        Calculate the ellipsometric values (psi, delta) of this model

        Parameters
        ----------
        wavelength_aoi : array-like
            An array of shape (N, 2) corresponding to the wavelengths (nm) and
            angle of incidences (deg) the ellipsometric measurements were
            performed at.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        """
        if p is not None:
            self.parameters.pvals = np.array(p)

        wavelength, aois = wavelength_aoi.T
        psi = np.zeros_like(wavelength)
        delta = np.zeros_like(wavelength)

        unique_wavelengths = np.unique(wavelength)
        for wav in unique_wavelengths:
            idx = np.where(wavelength == wav)
            aoi = aois[idx]
            self.structure.wavelength = wav
            _psi, _delta = Delta_Psi_TMM(
                AOI=aoi,
                layers=self.structure.slabs()[..., :4],
                wavelength=wav,
                delta_offset=self.delta_offset.value,
                reflect_delta=self._flip_delta,
            )
            psi[idx] = _psi
            delta[idx] = _delta

        return psi, delta

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
        if not isinstance(structure, StructureSE):
            raise ValueError("structure must be a StructureSE instance")

        self._structure = structure

        p = Parameters(name="instrument parameters")
        p.extend([self.delta_offset])

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
