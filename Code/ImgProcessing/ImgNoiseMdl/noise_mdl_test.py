
import os
import numpy as np
import matplotlib.pyplot as plt

R = np.random.poisson(10, size=1000)

plt.hist(R, bins=100)
plt.show()
