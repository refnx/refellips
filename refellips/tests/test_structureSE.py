import os.path
import numpy as np

from refellips import RI, DataSE, ReflectModelSE, ObjectiveSE


def test_cauchy_against_wvase():
    # Check the cauchy model behaves as expected
    pth = os.path.dirname(os.path.abspath(__file__))

    A = 1.47
    B = 0.00495
    C = 0

    _f = os.path.join(pth, "../materials/silicon.csv")
    si = RI(_f)

    _f = os.path.join(pth, "../materials/void.csv")
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
