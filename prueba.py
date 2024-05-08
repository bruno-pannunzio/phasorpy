#%%
from phasorpy.io import read_fbd
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center, phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import fractions_from_phasor, two_fractions_from_phasor

data_path = 'test_data/FBDfiles-DIVER/BUENOS/convallaria_000$EI0S.fbd'
calibration_path = 'test_data/FBDfiles-DIVER/BUENOS/RH110CALIBRATION_000$EI0S.fbd'

data_signal = read_fbd(data_path, frame=-1,channel=0, keepdims=False)
calibration_signal = read_fbd(calibration_path, frame=-1,channel=0, keepdims=False)

mean, real, imag = phasor_from_signal(data_signal)
mean_calib, real_calib, imag_calib = phasor_from_signal(calibration_signal)

reference_real, reference_imag = phasor_center(real_calib, imag_calib)
real, imag = phasor_calibrate(real, imag, reference_real, reference_imag, frequency=80, lifetime=4)
plot = PhasorPlot(frequency=80)
plot.hist2d(real, imag, cmap = 'plasma', bins = 300)
frequency = 80.0
components_lifetimes = [8.0, 0.5, 1.5]
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
plot.plot(components_real, components_imag, fmt =  'o-')
fractions = fractions_from_phasor(real, imag, components_real, components_imag)
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(
    fractions[:,:,0].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='8 ns',
)
ax.hist(
    fractions[:,:,1].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='0.5 ns',
)
ax.hist(
    fractions[:,:,2].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='2 ns',
)
ax.set_title('Histograms of fractions of 3 components')
ax.set_xlabel('Fraction')
ax.set_ylabel('Counts')
ax.legend()
plt.tight_layout()
plt.show()



#%%
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import new_fractions_from_phasor, two_fractions_from_phasor
import numpy as np
frequency = 80.0
components_lifetimes = [8.0, 1.0, 3.0]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.10, 0.20, 0.70],[0.10, 0.20, 0.70]]
    )
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
# plot = PhasorPlot(frequency=frequency, title = 'Phasor lying on the line between components')
# plot.plot(components_real, components_imag, fmt= 'o-')
# plot.plot(real, imag)
# plot.show()
# real  = np.expand_dims(real, axis=-1).repeat(3, axis=-1)
# imag  = np.expand_dims(imag, axis=-1).repeat(3, axis=-1)
# print('Test 3 components')
# fractions = fractions_from_phasor(real, imag, components_real, components_imag)
# fractions = four_fractions_from_phasor(real, imag, components_real, components_imag, components=3)
# print(fractions)
print('Test 2 components')
fractions = new_fractions_from_phasor(real[0], imag[0], components_real[:2], components_imag[:2], components=2)
print ('Fraction from first component: ', fractions[0])
print ('Fraction from second component: ', fractions[1])
fractions = two_fractions_from_phasor(real, imag, components_real[:2], components_imag[:2])
print('two fractions phasor function: ')
print ('Fraction from first component: ', fractions[0])
print ('Fraction from second component: ', fractions[1])

# %%
# PRUEBA 4 COMPONENTES
from phasorpy.io import read_fbd, read_ptu
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center, phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import fractions_from_phasor, two_fractions_from_phasor
import numpy as np

# data_path = 'test_data/FBDfiles-DIVER/BUENOS/convallaria_000$EI0S.fbd'
# calibration_path = 'test_data/FBDfiles-DIVER/BUENOS/RH110CALIBRATION_000$EI0S.fbd'

data_path = 'test_data/flim/calibration/convularia/Ex780p64_FB_convularia_c400_000$CA0S.fbd'
calibration_path = 'test_data/flim/calibration/convularia/Ex780p64_FB_coumarin_c100_000$CA0S.fbd'

data_signal = read_fbd(data_path, frame=-1,channel=0, keepdims=False)
calibration_signal = read_fbd(calibration_path, frame=-1,channel=0, keepdims=False)

mean, real, imag = phasor_from_signal(data_signal, harmonic=[1,2])
mean_calib, real_calib, imag_calib = phasor_from_signal(calibration_signal, harmonic=[1,2])

#%%
reference_real_first, reference_imag_first = phasor_center(real_calib[0], imag_calib[0])
reference_real_second, reference_imag_second = phasor_center(real_calib[1], imag_calib[1])
# real_first, imag_first = phasor_calibrate(real[0], imag[0], reference_real_first, reference_imag_first, frequency=78, lifetime=4)
# real_second, imag_second = phasor_calibrate(real[1], imag[1], reference_real_second, reference_imag_second, frequency=78*2, lifetime=4)
real_first, imag_first = phasor_calibrate(real[0], imag[0], reference_real_first, reference_imag_first, frequency=80, lifetime=2.5)
real_second, imag_second = phasor_calibrate(real[1], imag[1], reference_real_second, reference_imag_second, frequency=80*2, lifetime=2.5)
#%%
plot = PhasorPlot(frequency=80)
plot.hist2d(real_first, imag_first, cmap = 'plasma', bins = 300)
#%%
plot = PhasorPlot(frequency=160, allquadrants=True)
plot.hist2d(real[0], imag[0], cmap = 'plasma', bins = 300)
#%%
#PRUEBA 4 COMPONENTES SINTETICO
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import new_fractions_from_phasor
import numpy as np

components_lifetimes = [0.3, 1.0, 3.0, 8.0]
frequency = [80.0, 160.0]
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)

# plot = PhasorPlot(frequency=80.0)
# plot.plot(components_real, components_imag, fmt =  'o-')
# real, imag = phasor_from_lifetime(
#         frequency, components_lifetimes, [0.10, 0.20, 0.45, 0.25],
#     )

# real, imag = phasor_from_lifetime(
#         frequency, components_lifetimes, [[0.10, 0.20, 0.45, 0.25],[0.15, 0.20, 0.40, 0.25]],
#     )
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.10, 0.20, 0.45, 0.25],[0.15, 0.20, 0.40, 0.25], [0.15, 0.20, 0.40, 0.25]],
    )

real  = np.expand_dims(real, axis=-1).repeat(3, axis=-1)
imag  = np.expand_dims(imag, axis=-1).repeat(3, axis=-1)


# plot.plot(real[0], imag[0], c = 'blue')
# plot.plot(real[1], imag[1], c = 'red')

fractions = new_fractions_from_phasor(real, imag, components_real, components_imag)
print(fractions)
print(fractions[0].shape)
print(fractions[0])
# %%
