import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

v  = np.linspace(-3,3,500)
y = stats.norm.pdf(v,0,1)
plt.plot(v, stats.norm.pdf(v,0,1),label='Normal distribution')
yt = np.array([ 0 if z < -1.5 or z > 1.5 else e for e,z in zip(y,v) ])
plt.ylim(0,0.5)
plt.plot(v, yt, label="Truncated normal distribution using 1.5 as hyperparameter")
plt.legend(loc = 'upper center')
plt.show()