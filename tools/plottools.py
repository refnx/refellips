# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 08:40:16 2021

@author: Isaac
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_ellipsdata (ax, data=None, model=None, objective=None, xaxis='aoi', plot_labels=True, legend=True):
    """
    Plots delta and psi values as a function of wavelength or angle of incidence.
    Compatible with VASE.
    
    Must pass an axis object for plot_ellipsdata to plot on. To create an axis
    object:
        
        fig, ax = plt.subplots()

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis object on which delta and psi values are plotted.
    data : refellips.dataSE.DataSE, optional
        Data to plot. The default is None.
    model : refellips.reflect_modelSE.ReflectModelSE, optional
        Model to plot. If the model is provided, data must also be provided.
        The default is None.
    objective : refellips.objectiveSE.ObjectiveSE, optional
        Objective (containing model and data) to plot. If the objective is provided,
        neither model or data should be provided. The default is None.
    xaxis : String, optional
        Either 'aoi' or 'wavelength'. The default is 'aoi'.
    plot_labels : Bool, optional
        Whether to plot axis labels. The default is True.
    legend : Bool, optional
        Whether to plot the legend. The default is True.

    Returns
    -------
    None.

    """
    
    if objective != None:
        assert data == None and model == None, "If objective is supplied, model and data should not be passed"
        data = objective.data
        model = objective.model
    elif model != None:
        assert data != None, "If you supply a model, you must also supply data"
    else:
        assert data != None, "must supply at least one of data, model or objective"
        
    assert xaxis == 'aoi' or xaxis == 'wavelength', "xaxis must be 'aoi' or 'wavelength"
        
    axt = ax.twinx()
    data.mask = np.ones_like(data.mask, dtype=bool)

    if xaxis == 'aoi':
        unique_wavs = np.unique(data.wav)
        aois = np.linspace(np.min(data.aoi)-5, np.max(data.aoi)+5)
        x = data.aoi
        xlab = 'AOI, °'

        if model != None:
            for wav in unique_wavs:
                model.wav=wav
                psis, deltas = model(aois)
                ax.plot(aois, psis, color='r')
                axt.plot(aois, deltas, color='b')

    elif xaxis == 'wavelength':
        unique_aois = np.unique(data.aoi)
        wavs = np.linspace(np.min(data.wav)-50, np.max(data.wav)+50)
        x = data.wav
        
        if model != None:
            for u in unique_aois:
                psis = []
                deltas = []
                for wav in wavs:
                    model.wav = wav
                    psi, delta = model([u])
                    psis.append(psi)
                    deltas.append(delta)
                
            ax.plot(wavs, psis, color='r')
            axt.plot(wavs, deltas, color='b')
            xlab = 'Wavelength, nm'

    p=ax.scatter(x, data.psi, color='r')
    d=axt.scatter(x, data.delta, color='b')

    ax.legend(handles=[p,d], labels=['Psi', 'Delta'])
        
    if plot_labels:
        ax.set(ylabel='Psi', xlabel=xlab)
        axt.set(ylabel='Delta')


def plot_structure(ax, objective=None, structure=None, reverse_structure=False, plot_labels=True):
    """
    Plots refractive index as a function of distance from the substrate.
    Compatible with VASE.
    
    Must pass an axis object for plot_structure to plot on. To create an axis
    object:
        
        fig, ax = plt.subplots()

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis object on which refractive index values are plotted.
    objective : refellips.objectiveSE.ObjectiveSE, optional
        Objective (containing model and data) to plot. If the objective is provided,
        structure should not be provided. The default is None.
    structure : refnx.reflect.structure.Structure
        Structure (which represents the interface) to be plotted. If structure is provided,
        objective should not be provided. The default is None.
    reverse_structure : bool
        Reverses the structure describing the interface - i.e. the order of the slabs is reversed.
    plot_labels : Bool, optional
        Whether to plot axis labels. The default is True.

    Returns
    -------
    None.

    """
    if objective != None:
        assert structure == None, "you must supply either an objective or structure, not both"
        structure = objective.model.structure
        wavelengths = np.unique(objective.data.wav)
        model = objective.model
    else: 
        assert structure != None, "you must supply either an objective or structure"
        wavelengths = [658]
        model = False
        
    if len(wavelengths) > 1:
        colors = plt.cm.viridis(np.linspace(0,1,len(wavelengths)))
        alpha=0.5
    else:
        colors=['k']
        alpha=1
    
    
    structure.reverse_structure = reverse_structure

    for wav, col in zip(wavelengths, colors):
        for x in structure:
            if model:
                model.wav = wav
            else:
                x.sld.set_wav = wav

        ax.plot(*structure.sld_profile(), color=col, alpha=alpha, label=f'{wav} nm')

    structure.reverse_structure = False

    if plot_labels:
        ax.set(ylabel='Refractive index', xlabel='Distance from substrate (Å)')
        ax.legend(ncol=2, fontsize=7, frameon=False)