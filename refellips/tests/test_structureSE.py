import os.path
from pathlib import Path
from os.path import join as pjoin
import glob
import numpy as np
from numpy.testing import assert_allclose
from refnx._lib import flatten
from refnx.reflect import Linear

import refellips
from refellips import (
    RI,
    DataSE,
    ReflectModelSE,
    ObjectiveSE,
    Cauchy,
    Sellmeier,
    Lorentz,
    TaucLorentz,
    Gauss,
    MixedSlabSE,
    load_material,
)


pth = Path(os.path.dirname(os.path.abspath(refellips.__file__)))


def test_cauchy_against_wvase():
    # Check the cauchy model behaves as expected
    A = 1.47
    B = 0.00495
    C = 0

    cauchy = Cauchy(A=A, B=B, C=C)

    _f = pth / "tests" / "Cauchynk_fromWVASE.txt"
    wvase_output = np.loadtxt(_f)
    wavs = wvase_output[:, 0]
    refin = A + B / ((wavs / 1000) ** 2) + C / ((wavs / 1000) ** 4)

    refellips_RI = np.real(cauchy.complex(wavs))

    assert_allclose(refellips_RI, wvase_output[:, 1], rtol=0.000001)
    assert_allclose(refellips_RI, refin, rtol=0.000001)


def test_RI_from_array():
    A = 1.47
    B = 0.00495
    C = 0

    wavs = np.arange(1, 101.0)
    ri_in = A + B / ((wavs) ** 2) + C / ((wavs) ** 4)
    ec_in = np.linspace(0.01, 0.2, 100)

    _f = RI((wavs, ri_in, ec_in))
    assert_allclose(_f._wav[0], 1000.0)
    ri = _f.complex(2000.0)
    assert_allclose(ri, complex(ri_in[1], ec_in[1]))


def test_sellmeier_against_CompleteEase():
    # Check the Sellmeier model behaves as expected
    A = 1
    E = 2
    P = 0.01
    Einf = 1
    s = Sellmeier(A, E, P, Einf)

    _f = pth / "tests" / "Sellmeiernk_fromCompleteEase.txt"
    wvase_output = np.loadtxt(_f)
    wavs = wvase_output[:, 0]

    refellips_RI_n = np.real(s.complex(wavs))

    assert_allclose(refellips_RI_n, wvase_output[:, 1], rtol=6e-7)


def test_lorentz_against_wvase():
    # Check the Lorentz model behaves as expected
    A = [5, 10]
    B = [0.25, 0.5]
    E = [2, 4]
    Einf = 2
    lo = Lorentz(A, B, E, Einf)
    assert len(lo.Am) == 2

    _f = pth / "tests" / "Lorentznk_fromWVASE.txt"
    wvase_output = np.loadtxt(_f)
    wavs = wvase_output[:, 0]

    refellips_RI_k = np.imag(lo.complex(wavs))
    refellips_RI_n = np.real(lo.complex(wavs))

    assert_allclose(refellips_RI_n, wvase_output[:, 1], rtol=0.0016)
    assert_allclose(refellips_RI_k, wvase_output[:, 2], rtol=0.0019)


def test_gauss_against_CompleteEase():
    # Check the Gauss model behaves as expected
    A = [0.5]
    B = [1]
    E = [2.5]
    Einf = 1
    g = Gauss(A, B, E, Einf)
    assert len(g.Am) == 1

    _f = pth / "tests" / "Gaussnk_fromCompleteEase.txt"
    wvase_output = np.loadtxt(_f)
    wavs = wvase_output[:, 0]

    refellips_RI_k = np.imag(g.complex(wavs))
    refellips_RI_n = np.real(g.complex(wavs))

    assert_allclose(refellips_RI_n, wvase_output[:, 1], rtol=0.0013)
    assert_allclose(refellips_RI_k, wvase_output[:, 2], rtol=0.0047)


def test_Gaussian_multi_against_CompleteEase():
    # Check the Gauss model with multiple oscillators behaves as expected
    Am = [1.7, 0.5, 1.3]
    Br = [0.4, 3, 0.22]
    En = [0.12, 2.98, 1.6]
    Einf = 1
    g = Gauss(Am, Br, En, Einf)

    _f = pth / "tests" / "GaussianMulti_fromCompleteEase.txt"
    wvase_output = np.loadtxt(_f)
    energy = wvase_output[:, 0]

    refellips_e = g.epsilon(energy)
    refellips_e1 = np.real(refellips_e)
    refellips_e2 = np.imag(refellips_e)

    assert_allclose(refellips_e1, wvase_output[:, 1], rtol=0.0097)
    assert_allclose(refellips_e2, wvase_output[:, 2], rtol=0.0067)


def test_TaucLorentz_against_CompleteEase():
    # Check the Gauss model behaves as expected
    Am = [22, 154]
    C = [0.9, 1.6]
    En = [1.2, 3.4]
    Eg = 1.1
    Einf = 1
    tl = TaucLorentz(Am, C, En, Eg, Einf)

    _f = pth / "tests" / "TaucLorentz_fromCompleteEase.txt"
    wvase_output = np.loadtxt(_f)
    energy = wvase_output[:, 0]

    refellips_e = tl.epsilon(energy)
    refellips_e1 = np.real(refellips_e)
    refellips_e2 = np.imag(refellips_e)

    assert_allclose(refellips_e1, wvase_output[:, 1], rtol=6.5e-6)
    assert_allclose(refellips_e2, wvase_output[:, 2], rtol=4.5e-6)


def test_dispersions_are_loadable():
    # test that all the bundled dispersion curves are loadable
    materials = glob.glob(str(pth / "materials" / "*.csv"))
    for material in materials:
        _f = RI(material)
        assert len(_f._wav) > 1


def test_mixedslab():
    a = load_material("water")
    b = Cauchy(A=1.47, B=0.00495)

    slab = MixedSlabSE(10, a, b, 0.5, 3)
    assert b.A in flatten(slab.parameters)
    assert slab.slabs().shape == (1, 5)
    assert_allclose(slab.vf_B.value, 0.5)

    a.wavelength = 400
    b.wavelength = 400
    riac = a.complex(400)
    ribc = b.complex(400)

    slab.vf_B.value = 0
    assert_allclose(slab.slabs()[0, 1], np.real(riac))
    assert_allclose(slab.slabs()[0, 2], np.imag(riac))

    slab.vf_B.value = 0.25
    overall = np.sqrt(0.75 * riac**2 + 0.25 * ribc**2)
    assert_allclose(slab.slabs()[0, 1], np.real(overall))
    assert_allclose(slab.slabs()[0, 2], np.imag(overall))


def test_micro_slicing():
    i = Linear()
    air = load_material("air")
    si = load_material("silicon")
    sio2 = load_material("silica")
    s = air | sio2(100, 10) | si(0, 3)
    s[1].interfaces = i
    assert len(s.slabs()) > 300
