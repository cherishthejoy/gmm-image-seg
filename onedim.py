# The whole thing about 1 dimensional Gaussian is that
# They can be explained as the intensity of a grayscale pixel 
# For example: the distribution of many pixel values across an image 
# can be modeled by 1D Gaussian

# So a grayscale image pixels can be divided into 2 Gaussians, depending 
# on the intensity value of the pixel, dark + bright regions


import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import log_loss


from sklearn.mixture import GaussianMixture as GMM


def gmm_log_likelihood(X, weights, means, covars):
    """
    Calculate log-likelihood for a 1D Gaussian Mixture Model
    
    Parameters:
    X: array of data points
    weights: array of mixture weights
    means: array of component means
    covars: array of component variances
    """
    n_components = len(weights)
    n_samples = len(X)
    
    # Calculate probability of each point under each component
    # Shape: (n_samples, n_components)
    prob_components = np.zeros((n_samples, n_components))
    
    for i in range(n_components):
        prob_components[:, i] = weights[i] * stats.norm.pdf(
            X, means[i], np.sqrt(covars[i])
        )
    
    # Sum across components to get mixture probability for each point
    mixture_prob = np.sum(prob_components, axis=1)
    
    # Take log and sum to get log-likelihood
    log_likelihood = np.sum(np.log(mixture_prob))
    
    return log_likelihood

# Apparently i have to use this in new codes
rng = np.random.default_rng()

x = np.concatenate((rng.normal(5, 5, 1000), rng.normal(10, 2, 1000)))
f = x.reshape(-1, 1)


gmm = GMM(n_components=3, covariance_type='full')
gmm.fit(f)

weights = gmm.weights_
means = gmm.means_
covars = gmm.covariances_


print(gmm.n_iter_)
print(gmm.converged_)

x_axis = x
x_axis.sort()

plt.style.use('dark_background')
plt.hist(f, bins=100, histtype='bar', density=True, ec='white', alpha=1.0)
plt.plot(x_axis, weights[0] * stats.norm.pdf(x_axis, means[0], np.sqrt(covars[0])).ravel(), c = 'red')
plt.plot(x_axis, weights[1] * stats.norm.pdf(x_axis, means[1], np.sqrt(covars[1])).ravel(), c = 'green')
plt.plot(x_axis, weights[2] * stats.norm.pdf(x_axis, means[2], np.sqrt(covars[2])).ravel(), c = 'blue')


print(weights[0] + weights[1] + weights[2]) # This sums up to 1


# These 2 pretty much do the same thing
log_likelihood = gmm.score(f) * len(f)
print(log_likelihood)

like = gmm_log_likelihood(x_axis, weights, means, covars)
print(like)

plt.grid()
plt.show()
plt.plot(x)








