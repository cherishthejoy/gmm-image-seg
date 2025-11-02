import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM
from sklearn.datasets import make_blobs
from matplotlib.patches import Arc


X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting


rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))


gmm = GMM(n_components=4, covariance_type='full', random_state=42)

gmm.fit(X_stretched)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse or Arc
    for nsig in range(1, 4):
        ax.add_patch(Arc(xy=position, width=nsig * width, height=nsig * height, angle=angle, ls='-.', **kwargs))
        
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, marker="x", s=20, cmap='cividis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=20, marker="x", zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)   

plt.style.use('dark_background')
plt.figure(figsize=(8, 6))
plot_gmm(gmm, X_stretched)
plt.title('GMM Clustering with Ellipses')
plt.show()