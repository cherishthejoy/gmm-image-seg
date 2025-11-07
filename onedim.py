import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture as GMM

# Apparently I have to use this in new codes
rng = np.random.default_rng()

# Here we pretty much say the ground truth of the pi values for each clusters 
# [1000 / 2000] and [1000 / 2000]
x = np.concatenate((rng.normal(1, 2, 1000), rng.normal(9, 3, 1000)))
f = x.reshape(-1, 1)

gmm = GMM(n_components=2, covariance_type='full')
gmm.fit(f)

weights = gmm.weights_
means = gmm.means_
covars = gmm.covariances_

x_axis = np.sort(x)

plt.style.use('dark_background')
plt.hist(f, bins=100, histtype='bar', density=True, ec='white', alpha=1.0)
plt.plot(x_axis, weights[0] * stats.norm.pdf(x_axis, means[0], np.sqrt(covars[0])).ravel(), c = 'red')
plt.plot(x_axis, weights[1] * stats.norm.pdf(x_axis, means[1], np.sqrt(covars[1])).ravel(), c = 'green')

plt.grid()
plt.show()








