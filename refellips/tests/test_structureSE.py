import os.path
from os.path import join as pjoin
import glob
import numpy as np
from numpy.testing import assert_allclose

import refellips
from refellips import RI, DataSE, ReflectModelSE, ObjectiveSE


def test_cauchy_against_wvase():
    # Check the cauchy model behaves as expected
    A = 1.47
    B = 0.00495
    C = 0

    cauchy = RI(A=A, B=B, C=C)

    pth = os.path.dirname(os.path.abspath(__file__))
    _f = pjoin(pth, "Cauchynk_fromWVASE.txt")
    wvase_output = np.loadtxt(_f)
    wavs = wvase_output[:, 0]
    refin = A + B / ((wavs / 1000) ** 2) + C / ((wavs / 1000) ** 4)

    refellips_RI = [cauchy.complex(wav).real for wav in wavs]

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


def test_dispersions_are_loadable():
    # test that all the bundled dispersion curves are loadable
    pth = os.path.dirname(os.path.abspath(refellips.__file__))
    pth = pjoin(pth, "materials")
    materials = glob.glob(pjoin(pth, "*.csv"))
    for material in materials:
        _f = RI(material)
        assert len(_f._wav) > 1
