import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM
from sklearn.datasets import make_blobs
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

n_samples = 300

rng = np.random.default_rng()
shifted_gaussian = rng.standard_normal((n_samples, 2)) + np.array([20, 20])

C = np.array([[0.0, -0.7], [3.5, 0.7]])
stretched_gaussian = np.dot(rng.standard_normal((n_samples, 2)), C)

D = np.array([[0.0, 0.5], [0.5, 0.7]])
third_gaussian = np.dot(rng.standard_normal((n_samples, 2)), D) + np.array([-10, 30])

X_train = np.vstack([shifted_gaussian, stretched_gaussian, third_gaussian])

gnn = GMM(n_components=3, covariance_type="full")
gnn.fit(X_train)

print(gnn.n_iter_)
print(gnn.score(X_train) * len(X_train))

max_iter = 50
log_likelihoods = []

for n_iter in range(1, max_iter + 1):
    gmm = GMM(n_components=3, covariance_type="full", max_iter=2, n_init=1, random_state=42)
    gmm.fit(X_train)
    log_likelihoods.append(gmm.score(X_train) * len(X_train))

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_iter + 1), log_likelihoods, marker='o', markersize=4)
plt.xlabel('EM Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs EM Iterations')
plt.grid(True, alpha=0.3)
plt.show()

x = np.linspace(-40.0, 50.0, 200)
y = np.linspace(-40.0, 50.0, 200)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gnn.score_samples(XX)
Z = Z.reshape(X.shape)

plt.style.use('dark_background')

fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), 
                levels=np.logspace(0, 3, 10),
                colors='white')

plt.clabel(CS, fmt=fmt, fontsize=10)

plt.scatter(X_train[:, 0], X_train[:, 1], 0.8, color='white')
plt.title("Negative log-likelihood predicted by a GMM")
plt.gca().set_aspect('equal')
plt.show()

