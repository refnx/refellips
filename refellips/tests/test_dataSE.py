import numpy as np
import os.path
from pathlib import Path

from refellips.dataSE import open_EP4file, DataSE

pth = Path(os.path.dirname(os.path.abspath(__file__)))


def test_multiple_areas():
    data = open_EP4file(pth / "post synthesis.dat")
    assert len(data) == 32
    assert "X pos" in data[0].metadata
    for d in data:
        np.testing.assert_allclose(len(d), 5)

    data = open_EP4file(pth / "19-1-1.dat")
    assert isinstance(data, DataSE)
    assert "Y pos" in data.metadata

    data = open_EP4file(pth / "15-1-1.dat")
    assert isinstance(data, DataSE)
    assert "Y pos" in data.metadata
    assert len(data) == 11
    np.testing.assert_allclose(data.psi[-1], 12.72666667)
