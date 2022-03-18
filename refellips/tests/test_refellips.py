import numpy as np
import os.path
from os.path import join as pjoin
from refellips import (
    RI,
    Cauchy,
    DataSE,
    ReflectModelSE,
    ObjectiveSE,
    SlabSE,
    StructureSE,
)
from numpy.testing import assert_allclose

pth = os.path.dirname(os.path.abspath(__file__))


def test_bare_against_wvase1():
    # Check bare interface ellipsometry calculation against wvase
    dname = pjoin(pth, "TestData_bareSI.txt")
    data = DataSE(data=dname)

    _f = pjoin(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = pjoin(pth, "../materials/void.csv")
    void = RI(_f)

    struc = void() | si()
    assert isinstance(struc[0], SlabSE)
    assert isinstance(struc, StructureSE)

    model = ReflectModelSE(struc)

    wav, aois, psi_d, delta_d = data.data
    wavelength_aoi = np.c_[wav, aois]
    psi, delta = model(wavelength_aoi)

    assert_allclose(psi, psi_d, rtol=0.002)
    assert_allclose(delta, delta_d, rtol=0.003)

    _ = struc.ri_profile()


def test_Cauchy_against_wvase2():
    # Check Cauchy model against wvase
    A = 1.47
    B = 0.00495
    C = 0

    cauchy = Cauchy(A=A, B=B, C=C)

    wvase_output = np.loadtxt(pjoin(pth, "Cauchynk_fromWVASE.txt"))
    wavs = wvase_output[:, 0]

    refin = A + B / ((wavs / 1000) ** 2) + C / ((wavs / 1000) ** 4)

    refellips_RI = cauchy.complex(wavs)

    assert_allclose(refellips_RI, refin)
    assert_allclose(refellips_RI, wvase_output[:, 1], rtol=1e-6)


def test_refellips_against_wvase3():
    # Three layer stack(cauchy & from file) compared to wvase
    # Also tests behaviour of Delta around 180°
    dname = os.path.join(pth, "TestData_cauchy.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/void.csv")
    void = RI(_f)

    cauchy = Cauchy(A=1.47, B=0.00495, C=0)
    struc = void() | cauchy(1000) | si()
    assert isinstance(struc, StructureSE)

    model = ReflectModelSE(struc)
    model._flip_delta = True

    wav, aois, psi_d, delta_d = data.data
    wavelength_aoi = np.c_[wav, aois]
    psi, delta = model(wavelength_aoi)

    # assert_allclose(psi, psi_d, rtol=0.0005)
    assert_allclose(delta, delta_d, rtol=0.003)
