import numpy as np
import os.path
from refellips import RI, DataSE, ReflectModelSE, ObjectiveSE
from numpy.testing import assert_allclose


def test_bare_against_wvase():
    # Check bare interface ellipsometry calcualtion against wvase
    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "TestData_bareSI.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/void.csv")
    void = RI(_f)

    struc = void() | si()
    model = ReflectModelSE(struc, wavelength=658)

    for dat in data.unique_wavelength_data():
        wav, aois, psi_d, delta_d = dat
        model.wav = wav
        psi, delta = model(aois)
        assert_allclose(psi, psi_d, rtol=0.002)
        assert_allclose(delta, delta_d, rtol=0.003)


def test_refellips_against_wvase3():
    # Three layer stack(cauchy & from file) compared to wvase
    # Also tests behaviour of Delta around 180°

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "TestData_cauchy.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/void.csv")
    void = RI(_f)

    cauchy = RI(A=1.47, B=0.00495, C=0)
    struc = void() | cauchy(1000) | si()
    model = ReflectModelSE(struc, wavelength=658)
    model._flip_delta = (
        True  # This will be automatically set when analysing data
    )

    for dat in data.unique_wavelength_data():
        wav, aois, psi_d, delta_d = dat
        model.wav = wav
        psi, delta = model(aois)
        assert_allclose(psi, psi_d, rtol=0.0005)
        assert_allclose(delta, delta_d, rtol=0.003)
