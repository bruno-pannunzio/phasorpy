#%%
from phasorpy.io import read_fbd
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center, phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import multiple_components_from_phasor
#%%
frequency = 80.0
components_lifetimes = [8.0, 3.0, 1.0]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [0.10, 0.20, 0.70]
    )
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
plot = PhasorPlot(frequency=frequency, title = 'Phasor lying on the line between components')
plot.plot(components_real, components_imag, fmt= 'o-')
plot.plot(real, imag)
plot.show()
fraction_from_first_component, fraction_from_second_component = multiple_components_from_phasor(real, imag, components_real, components_imag)
print ('Fraction from first component: ', fraction_from_first_component)
print ('Fraction from second component: ', fraction_from_second_component) 






#%%
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
# %%
