import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM

img = np.array(Image.open("fruits.jpg"))
rows, cols, ch = img.shape

img2 = img.reshape((-1, 3))

max_iter = 10
log_likelihoods = []

gmm = GMM(n_components=3, covariance_type="full", warm_start=True, 
            max_iter=1, n_init=1, random_state=42)

for n_iter in range(max_iter):
    gmm.fit(img2)
    ll = gmm.score(img2) * len(img2)
    log_likelihoods.append(ll)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_iter + 1), log_likelihoods, marker='o', markersize=4)
plt.xlabel('EM Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs EM Iterations')
plt.grid(True, alpha=0.3)
plt.show()