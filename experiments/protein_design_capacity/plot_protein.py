import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

x = np.loadtxt("results/output.txt", delimiter=",")

# sort x according to the first column
x = x[x[:,0].argsort()]

BK = 1.
plt.loglog(x[:,0], BK*x[:,1]/3050, 'ko-', label = 'theorerical error')
plt.loglog(x[:,0], 1/x[:,0], 'ro-', label = '1/T')
plt.grid()
plt.legend()
plt.show()
