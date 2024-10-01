import matplotlib.pyplot as plt
import numpy as np

bias_on = np.array([0, 20, 20, 40, 25])
bias_off = np.array([0, 20, 40, 20, 25])
SNR = []
PNR = [0.695, 0.291, 0.483, 0.823, 0.876]



plt.plot(bias_on-bias_off, PNR, "k+")
plt.show()