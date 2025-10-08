import numpy as np
import os.path
from pathlib import Path
from numpy.testing import assert_allclose

from refellips.dataSE import (
    DataSE,
    open_EP4file,
    open_FilmSenseFile,
    open_M2000file,
)

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


def test_filmsense_loader():
    # STANDARD DATA TEST
    std_fsdata = open_FilmSenseFile(pth / "Filmsense_staticTest.txt")

    std_bench = np.array(
        [
            [367.20, 26.46, 158.74],
            [449.64, 15.77, 173.19],
            [525.63, 12.64, 174.05],
            [593.58, 11.13, 174.54],
            [656.32, 10.25, 174.92],
            [732.22, 9.51, 175.37],
            [852.88, 8.76, 175.98],
            [946.73, 8.35, 176.36],
        ]
    ).T

    assert std_fsdata.metadata["numwvls"] == 8
    assert_allclose(std_fsdata.aoi, std_fsdata.metadata["nomAOI"])
    assert_allclose(std_bench[0], std_fsdata.wavelength, rtol=1e-3)
    assert_allclose(std_bench[1], std_fsdata.psi, rtol=1e-3)
    assert_allclose(std_bench[2], std_fsdata.delta, rtol=1e-3)

    # DYNAMIC DATA TEST
    dyn_fsdata = open_FilmSenseFile(pth / "Filmsense_kineticTest.txt")

    bench_times = [4.95, 14.86, 24.77, 34.66, 44.57, 54.48, 64.38]

    times = list(dyn_fsdata.keys())

    assert_allclose(bench_times, times, rtol=1e-3)

    sing_dyn_fsdata = dyn_fsdata[times[2]]

    dyn_bench = np.array(
        [
            [367.20, 27.04, 148.91],
            [449.64, 16.09, 162.67],
            [525.63, 12.91, 163.83],
            [593.58, 11.36, 164.72],
            [656.32, 10.45, 165.54],
            [732.22, 9.68, 166.52],
            [852.88, 8.89, 167.94],
            [946.73, 8.45, 168.86],
        ]
    ).T

    assert sing_dyn_fsdata.metadata["numwvls"] == 8
    assert_allclose(sing_dyn_fsdata.aoi, sing_dyn_fsdata.metadata["nomAOI"])
    assert_allclose(dyn_bench[0], sing_dyn_fsdata.wavelength, rtol=1e-3)
    assert_allclose(dyn_bench[1], sing_dyn_fsdata.psi, rtol=1e-3)
    assert_allclose(dyn_bench[2], sing_dyn_fsdata.delta, rtol=1e-3)


def test_M2000_loader():
    # STANDARD DATA TEST
    std_wdata = open_M2000file(pth / "Woolam_staticTest.dat", dropdatapoints=40)

    std_bench = np.array(
        [
            [370.324, 70, 21.4553, 145.7525],
            [434.059, 70, 8.6922, 157.8022],
            [497.825, 70, 4.7986, 152.0333],
            [561.586, 70, 2.8442, 141.3818],
            [625.304, 70, 1.7848, 123.4001],
            [688.942, 70, 1.2906, 97.0818],
            [752.463, 70, 1.2037, 69.1318],
            [815.829, 70, 1.3271, 49.0608],
            [879.003, 70, 1.5043, 36.8692],
            [941.948, 70, 1.6879, 28.9926],
        ]
    ).T

    assert std_wdata.metadata["AcqTime"] == "30.013"

    assert_allclose(std_bench[0], std_wdata.wavelength, rtol=1e-3)
    assert_allclose(std_bench[1], std_wdata.aoi, rtol=1e-3)
    assert_allclose(std_bench[2], std_wdata.psi, rtol=1e-3)
    assert_allclose(std_bench[3], std_wdata.delta, rtol=1e-3)

    # DYNAMIC DATA TEST
    dyn_wdata = open_M2000file(
        pth / "Woolam_kineticTest.dat", take_every=2, dropdatapoints=40
    )

    bench_times = [1.4, 6.4, 11.5, 16.5]
    times = list(dyn_wdata.keys())

    assert_allclose(bench_times, times, rtol=1e-3)

    sing_dyn_wdata = dyn_wdata[times[2]]

    dyn_bench = np.array(
        [
            [370.324, 70.0, 30.3104, 121.2913],
            [434.059, 70.0, 20.2057, 130.5181],
            [497.825, 70.0, 16.3773, 132.2358],
            [561.586, 70.0, 14.2143, 133.8754],
            [625.304, 70.0, 12.9270, 136.1897],
            [688.942, 70.0, 11.7856, 137.8630],
            [752.463, 70.0, 11.0232, 139.8401],
            [815.829, 70.0, 10.4095, 141.3768],
            [879.003, 70.0, 9.9504, 143.1087],
            [941.948, 70.0, 9.5185, 144.6035],
        ]
    ).T

    assert sing_dyn_wdata.metadata["AcqTime"] == "0.978"
    assert_allclose(dyn_bench[0], sing_dyn_wdata.wavelength, rtol=1)
    assert_allclose(dyn_bench[1], sing_dyn_wdata.aoi, rtol=1e-3)
    assert_allclose(dyn_bench[2], sing_dyn_wdata.psi, rtol=1e-3)
    assert_allclose(dyn_bench[3], sing_dyn_wdata.delta, rtol=1e-3)
