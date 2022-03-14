import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
# sys.path.append('../')

import refellips
from refellips.structureSE import RI
from refellips.dataSE import DataSE
from refellips.reflect_modelSE import ReflectModelSE
from refellips.objectiveSE import ObjectiveSE



######
dname = 'TestData_bareSI.txt'
data = DataSE(data=dname)

si = RI('../materials/silicon.csv')
void = RI('../materials/void.csv')
struc = void() | si()
model = ReflectModelSE(struc)

fig, ax = plt.subplots()
axt = ax.twinx()
test_arr = []

for idx, wav in enumerate(np.unique(data.wavelength)):
    wavelength, aoi, d_psi, d_delta = list(data.unique_wavelength_data())[idx]
    psi, delta = model(np.c_[np.ones_like(aoi) * wavelength, aoi])

    ax.scatter(aoi, psi, facecolor='none', color='r', marker='s')
    p = ax.scatter(aoi, d_psi, facecolor='none', color='r')
    test_arr.append(np.abs(np.array(psi - d_psi) / d_psi) < 0.005)
    test_arr.append(np.abs(np.array(delta - d_delta) / d_delta) < 0.005)

    axt.scatter(aoi, delta, facecolor='none', color='b', marker='s')
    d = axt.scatter(aoi, d_delta, facecolor='none', color='b')

    ax.legend(handles=[p, d], labels=['Psi', 'Delta'])
    ax.set(ylabel='Psi', xlabel='AOI, °')
    axt.set(ylabel='Delta')

test_arr = np.array(test_arr)

if np.prod(test_arr, dtype=bool):
    print('Test 1 passed')
else:
    print('test 1 failed')

dname = 'TestData_cauchy.txt'
data = DataSE(data=dname)

si = RI('../materials/silicon.csv')
void = RI('../materials/void.csv')
cauchy = RI(A=1.47, B=0.00495, C=0)
struc = void() | cauchy(1000) | si()
model = ReflectModelSE(struc)
model._flip_delta = True  # This will be automatically set when analysing data

fig, ax = plt.subplots()
axt = ax.twinx()
test_arr = []

for idx, wav in enumerate(np.unique(data.wavelength)):
    wavelength, aoi, d_psi, d_delta = list(data.unique_wavelength_data())[idx]
    psi, delta = model(np.c_[np.ones_like(aoi) * wavelength, aoi])
    ax.scatter(aoi, psi, facecolor='none', color='r', marker='s')
    p = ax.scatter(aoi, d_psi, facecolor='none', color='r')
    test_arr.append(np.abs(np.array(psi - d_psi) / d_psi) < 0.005)
    test_arr.append(np.abs(np.array(delta - d_delta) / d_delta) < 0.005)

    axt.scatter(aoi, delta, facecolor='none', color='b', marker='s')
    d = axt.scatter(aoi, d_delta, facecolor='none', color='b')

    ax.legend(handles=[p, d], labels=['Psi', 'Delta'])
    ax.set(ylabel='Psi', xlabel='AOI, °')
    axt.set(ylabel='Delta')

test_arr = np.array(test_arr)

if np.prod(test_arr, dtype=bool):
    print('Test 2 passed')
else:
    print('test 2 failed')


data = DataSE('../../demos/WVASE_example_30nmSiO2_Water_MultiWavelength.txt')

si     = RI('../materials/silicon.csv')
sio2   = RI('../materials/silica.csv')
h2o    = RI('../materials/water.csv')

struc = h2o() | sio2(300) | si()
struc.solvent= h2o

model = ReflectModelSE(struc, delta_offset=0)
model._flip_delta = True

fig, ax = plt.subplots()
axt = ax.twinx()

wavelength, aoi, d_psi, d_delta = data.data
psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

ax.plot(wavelength, d_psi,ls='dotted', color='k', label='wvase', zorder=3)
axt.plot(wavelength, d_delta,ls='dotted', color='k', zorder=3)
ax.plot(wavelength, psi, color='r', label='refellips')
axt.plot(wavelength, delta, color='r')

ax.legend(frameon=False, loc = 'upper center')
ax.set(ylabel='Psi', xlabel='Wavelength (nm)')
axt.set(ylabel='Delta')


assert_allclose(psi, d_psi, rtol=5e-4)
assert_allclose(delta, d_delta, rtol=5e-4)


data = DataSE('../../demos/WVASE_example_2nmSiO2_20nmPNIPAM_50EMA_MultiWavelength_2.txt')

si      = RI('../materials/silicon.csv')
sio2    = RI('../materials/silica.csv')
polymer = RI(A=1.47, B=0.00495)
h2o     = RI(A=1.3242, B=0.003064)

polymer_layer = polymer(200)
polymer_layer.name = 'PNIPAM'
polymer_layer.vfsolv.setp(value=0.5)

struc = h2o() | polymer_layer | sio2(20) | si()
struc.solvent= h2o

model = ReflectModelSE(struc, delta_offset=0)
model._flip_delta = True

fig, ax = plt.subplots()
axt = ax.twinx()

wavelength, aoi, d_psi, d_delta = data.data
psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

ax.plot(wavelength, d_psi,ls='dotted', color='k', label='wvase', zorder=3)
axt.plot(wavelength, d_delta,ls='dotted', color='k', zorder=3)
ax.plot(wavelength, psi, color='r', label='refellips')
axt.plot(wavelength, delta, color='r')

ax.legend(frameon=False, loc = 'upper center')
ax.set(ylabel='Psi', xlabel='Wavelength (nm)')
axt.set(ylabel='Delta')

assert_allclose(psi, d_psi, rtol=6e-4)
assert_allclose(delta, d_delta, rtol=6e-4)



data = DataSE('../../demos/WVASE_example_2nmSiO2_117nmAu_MultiWavelength.txt')

si     = RI('../materials/silicon.csv')
sio2   = RI('../materials/silica.csv')
gold   = RI('../materials/gold.csv')
air    = RI('../materials/air.csv')

struc = air() | gold(1170) | sio2(20) | si()

model = ReflectModelSE(struc, delta_offset=0)
model._flip_delta = True

fig, ax = plt.subplots()
axt = ax.twinx()

wavelength, aoi, d_psi, d_delta = data.data
psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

ax.plot(wavelength, d_psi,ls='dotted', color='k', label='wvase', zorder=3)
axt.plot(wavelength, d_delta,ls='dotted', color='k', zorder=3)
ax.plot(wavelength, psi, color='r', label='refellips')
axt.plot(wavelength, delta, color='r')

ax.legend(frameon=False)
ax.set(ylabel='Psi', xlabel='Wavelength (nm)')
axt.set(ylabel='Delta')

assert_allclose(psi, d_psi, rtol=3e-4)
assert_allclose(delta, d_delta, rtol=3e-4)



data = DataSE('../../demos/WVASE_example_2nmSiO2_65nmAl2O3_MultiWavelength.txt')

si     = RI('../materials/silicon.csv')
sio2   = RI('../materials/silica.csv')
al2o3  = RI('../materials/aluminium_oxide.csv')
air    = RI('../materials/air.csv')

struc = air() | al2o3(3250) | sio2(100) | si()

model = ReflectModelSE(struc, delta_offset=0)
model._flip_delta = True

fig, ax = plt.subplots()
axt = ax.twinx()

wavelength, aoi, d_psi, d_delta = data.data
psi, delta = model(np.c_[wavelength, np.ones_like(wavelength) * aoi])

ax.plot(wavelength, d_psi,ls='dotted', color='k', label='wvase', zorder=3)
axt.plot(wavelength, d_delta,ls='dotted', color='k', zorder=3)
ax.plot(wavelength, psi, color='r', label='refellips')
axt.plot(wavelength, delta, color='r')

ax.legend(frameon=False, loc='lower center')
ax.set(ylabel='Psi', xlabel='Wavelength (nm)')
axt.set(ylabel='Delta')

assert_allclose(psi, d_psi, rtol=3e-4)
assert_allclose(delta, d_delta, rtol=3e-4)