import os.path
import numpy as np
from numpy.testing import assert_allclose

from refellips import RI, DataSE, ReflectModelSE, ObjectiveSE


def test_cauchy_against_wvase():
    # Check the cauchy model behaves as expected
    pth = os.path.dirname(os.path.abspath(__file__))

    A = 1.47
    B = 0.00495
    C = 0

    cauchy = RI(A=A, B=B, C=C)

    _f = os.path.join(pth, "Cauchynk_fromWVASE.txt")
    wvase_output = np.loadtxt(_f)
    wavs = wvase_output[:, 0]
    refin = A + B / ((wavs / 1000) ** 2) + C / ((wavs / 1000) ** 4)

    refellips_RI = [cauchy.complex(wav).real for wav in wavs]

    assert_allclose(refellips_RI, wvase_output[:, 1], rtol=0.000001)
    assert_allclose(refellips_RI, refin, rtol=0.000001)
