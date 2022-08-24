.. _faq_chapter:

====================================
Frequently Asked Questions
====================================

.. _github issues: https://github.com/refnx/refellips/issues
.. _refractiveindex.info: https://refractiveindex.info
.. _Markel: https://doi.org/10.1364/JOSAA.33.001244
.. _Humlicek: https://doi.org/10.1007/978-3-642-33956-1_3
.. _getting started: getting_started.ipynb#Saving-the-objective
.. _Gaussian oscillator: https://nbviewer.org/github/refnx/refellips/blob/master/demos/refellipsDemo_GaussianOscillator.ipynb
.. _Cauchy, Sellmeier: https://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Cauchy_and_related_empirical_dispersion_Formulae_for_Transparent_Materials.pdf
.. _Lorentz: https://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf

A list of common questions.


What's the best way to ask for help or submit a bug report?
-----------------------------------------------------------

If you have any questions about using *refellips* or calculations
performed by *refellips* please
`contact us <mailto:andyfaff+refellips@gmail.com>`_ or use the `GitHub Issues`_ tracker.
If you find a bug in the code or documentation, please use `GitHub Issues`_.


What are the 'fronting' and 'backing' media?
--------------------------------------------

The 'fronting' and 'backing' media are infinite. The 'fronting' medium carries
the incident beam of radiation, whilst the 'backing' medium will carry the
transmitted beam away from the interface. In short, the fronting media
is the medium that the radiation interacts with first, and the backing
media is the medium which the radiation interacts with last.

For example, consider a system with an oxidised silicon wafer where the
ambient material is air; the fronting media would be air and the backing
media would be silicon.


What formats/types of ellipsometry data does *refellips* handle?
----------------------------------------------------------------

*refellips* has the capability of loading data directly from both Accurion EP3
and EP4 ellipsometers, as well Horiba ellipsometers using the `open_EP4file`
and `open_HORIBAfile` functions, respectively.

Alternatively, users also have the option to load-in other datasets using
`DataSE`. Files loaded using `DataSE` must contain four columns (with header):
wavelength, angle of incidence, psi and delta.


Where do I find dispersion curves for a material?
-------------------------------------------------

*refellips* contains preloaded dispersion curves for select materials, which
are accessible by the `load_material` function. These materials are sourced
from `refractiveindex.info`_, and include air, a void, water,
dimethyl sulfoxide, silicon, silica, gold, aluminium oxide, polystyrene,
poly(N-isopropylacrylamide) (PNIPAM) and a material that represents a diffuse
polymer.

If required, users can download their own dispersion curves from
`refractiveindex.info`_ and load them into *refellips* using::

    my_material = RI("my_dispersion.csv")

The loaded file must contain at least two columns, assumed to be wavelength
(in microns) and refractive index.
If three columns are provided, the third is loaded as the extinction coefficient.
The *refellips* maintainers are happy to include additional dispersion curves
with the package; please ask if you'd like this to happen.

Alternatively, users have the option to choose from any of the in-built oscillator
functions to model the optical properties of their material: `Cauchy`, `Sellmeier`,
`Lorentz` and `Gauss`. Both the `Cauchy` and `Sellmeier` oscillators monotonically
decrease in refractive index with increasing wavelength and are therefore not
Kramers-Kronig consistent. These optical models are frequently used to model the
optical properties of transparent materials, however, the Sellmeier is more accurate
at higher wavelengths, i.e., the infra-red region. Users can specify `Cauchy` and
`Sellmeier` parameters for their material::

    my_cauchy_material = Cauchy(A=a, B=b, C=c)
    my_sellmeier_material = Sellmeier(Am, En, P, Einf)


Both the `Lorentz` and `Gaussian` functions are Kramers-Kronig consistent, and allow
users to implement multiple oscillators. `Lorentz` oscillators are typically employed
when working with materials above the fundamental band gap, describing well the optical
properties of transparent and weakly absorbing materials. `Gaussian` oscillators are
typically used for absorbing materials, where the complex component models the Gaussian
absorption and the real component is its Kramers-Kronig relation (a Hilbert transform).
Users can implement a one `Lorentz`, or two `Gaussian` oscillator model for their
material by::

    my_lorentz_material = Lorentz([Am], [Br], [En], Einf)
    my_gaussian_material = Gauss([Am_1, Am_2], [Br_1, Br_2], [En_1, En_2], Einf)

An example of the Gaussian oscillator is provided in the `Gaussian oscillator`_ notebook.
Parameter values for `Cauchy, Sellmeier`_ and `Lorentz`_ are provided by Horiba.
Cauchy parameters can also be found on `refractiveindex.info`_.

Alternatively, users can simply supply a refractive index (n) and extinction coefficient
(k) for a single wavelength measurement::

    my_material = RI([n, k])


What EMA methods does *refellips* provide?
------------------------------------------

*refellips* offers the three main methods of effective medium approximations
(EMA): linear, Maxwell Garnett and Bruggeman. All EMA calculations performed
in *refellips* are based on two-component mixing and done so using the
complex dielectric function, not refractive indices and extinction
coefficients.

For the examples below, :math:`\varepsilon_1` and :math:`f_1`
relate to the complex dielectric function and volume fraction of the lower
material (most commonly the host material) and :math:`\varepsilon_2` and
:math:`f_2` relate to the complex dielectric function and volume fraction
of the upper material (most commonly the inclusion material; e.g., solvent).
It is important to note that :math:`f_1 + f_2 = 1`.

For a linear EMA, the dielectric constant of the mixture is simply the sum
of the products of the substituent dielectric function and volume fraction
(Equation :eq:`linear`). *We hypothesise that the linear EMA will be
sufficient for most use cases.*

.. math::
    :label: linear

    \varepsilon_{\text{linear}} = f_1 \varepsilon_1 + f_2 \varepsilon_2

For the Maxwell Garnett and Bruggeman EMA methods, a depolarisation factor
(:math:`v`) is included to account for potential electric field screening
by anisotropic inclusions. When (:math:`v = 1/3`), Equation :eq:`mg` and
:eq:`bg` reduce down to the isotropic case, assuming all inclusions
are spherical in nature. We anticipate that only expert users will use
these EMA methods or alter the depolarisation factor.

The complex dielectric function for a mixed layer using the Maxwell-Garnett EMA
is determined using Equation :eq:`mg`,

.. math::
    :label: mg

    \varepsilon_{\text{MG}} = \varepsilon_1 \frac{\varepsilon_1 + (v f_1 + f_2)
            (\varepsilon_2 - \varepsilon_1)}
            {\varepsilon_1 + v f_1 (\varepsilon_2 - \varepsilon_1)}

The Bruggeman EMA method is employed using Equation :eq:`bg`,

.. math::
    :label: bg

    \varepsilon_{\text{BG}} = \frac{b +
                \sqrt{b^2 - 4 (v - 1) (e_1 e_2 v)}}
                {2(1 - v)}

where :math:`b = e_1 (f_1 - v) + e_2 (f_2 - v)`.

Further details surrounding these EMA methods and their derivations as
well as the depolarisation factor and anisotropy are explored by
both `Markel`_ and `Humlicek`_.

Can I save models/objectives to a file?
---------------------------------------
Assuming that you have a :class:`refellips.ReflectModelSE` or
:class:`refellips.ObjectiveSE` that you'd like to save to file,
the easiest way to do this is via serialisation to a Python pickle::

    import pickle
    # save
    with open('my_objective.pkl', 'wb+') as f:
        pickle.dump(objective, f)

    # load
    with open('my_objective.pkl', 'rb') as f:
        restored_objective = pickle.load(f)

The saved pickle files are in a binary format and are not human readable.
It may also be useful to save the representation, :code:`repr(objective)`.

Alternatively, modelled results can be exported into a `.csv` file. An
example of this is provided in `Getting started`_.
