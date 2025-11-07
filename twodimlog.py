import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

n_samples = 300

rng = np.random.default_rng()
c1 = rng.standard_normal((n_samples, 2)) + np.array([20, 20])

dot_two = np.array([[0.0, -0.7], [3.5, 0.7]])
c2 = np.dot(rng.standard_normal((n_samples, 2)), dot_two)

dot_three = np.array([[0.0, 0.5], [0.5, 0.7]])
c3 = np.dot(rng.standard_normal((n_samples, 2)), dot_three) + np.array([-10, 30])

X_train = np.vstack([c1, c2, c3])

gnn = GMM(n_components=3, covariance_type="full")
gnn.fit(X_train)

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

