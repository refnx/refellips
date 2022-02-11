import numpy as np
import os.path
from refellips import RI, DataSE, ReflectModelSE, ObjectiveSE


def test_bare_against_wvase():
    # Check bare interface ellipsometry calcualtion against wvase
    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "TestData_bareSI.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../../materials/void.csv")
    void = RI(_f)

    struc = void() | si()
    model = ReflectModelSE(struc, wavelength=658)

    test_arr = []

    for dat in data:
        model.wav = dat._current_wav
        aois = dat.aoi
        psi, delta = model(aois)
        test_arr.append(np.abs(np.array(psi - dat.psi) / dat.psi) < 0.01)
        test_arr.append(np.abs(np.array(delta - dat.delta) / dat.delta) < 0.01)

    assert np.all(test_arr)


def test_cauchy_against_wvase():
    # Check the cauchy model behaves as expected
    pth = os.path.dirname(os.path.abspath(__file__))

    A = 1.47
    B = 0.00495
    C = 0

    _f = os.path.join(pth, "../../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../../materials/void.csv")
    void = RI(_f)

    cauchy = RI(A=A, B=B, C=C)
    struc = void() | cauchy(1000) | si()
    model = ReflectModelSE(struc, wavelength=658)

    _f = os.path.join(pth, "Cauchynk_fromWVASE.txt")
    wvase_output = np.loadtxt(_f)
    wavs = wvase_output[:, 0]
    refin = A + B / ((wavs / 1000) ** 2) + C / ((wavs / 1000) ** 4)

    refellips_RI = []
    for wav in wavs:
        model.wav = wav
        refellips_RI.append(cauchy.real.value)

    passarr_wvase = (
        np.abs(refellips_RI - wvase_output[:, 1]) / wvase_output[:, 1] < 0.01
    )
    passarr_theory = np.abs(refellips_RI - refin) / refin < 0.01

    assert np.all(passarr_wvase)
    assert np.all(passarr_theory)


def test_refellips_against_wvase3():
    # Three layer stack(cauchy & from file) compared to wvase
    # Also tests behaviour of Delta around 180°

    pth = os.path.dirname(os.path.abspath(__file__))

    dname = os.path.join(pth, "TestData_cauchy.txt")
    data = DataSE(data=dname)

    _f = os.path.join(pth, "../../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../../materials/void.csv")
    void = RI(_f)

    cauchy = RI(A=1.47, B=0.00495, C=0)
    struc = void() | cauchy(1000) | si()
    model = ReflectModelSE(struc, wavelength=658)
    model._flip_delta = True  # This will be automatically set when analysing data

    test_arr = []

    for dat in data:
        model.wav = dat._current_wav
        aois = dat.aoi
        psi, delta = model(aois)
        test_arr.append(np.abs(np.array(psi - dat.psi) / dat.psi) < 0.01)
        test_arr.append(np.abs(np.array(delta - dat.delta) / dat.delta) < 0.01)

    assert np.all(test_arr)
