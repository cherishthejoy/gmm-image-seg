# Gaussian Mixture Model

### Clone and install the requirements

```
pip install -r requirements.txt

```

May or may not need to install the "Quarto" extension unless you use other IDE or text editors.

## Introduction

Gaussian Mixture Model is a "probabilistic" model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with "unknown" parameters.

Gaussian Mixture Model utilizes the idea of "soft-assignment" opposed to k-means clustering "hard-assignment" where each data point could belong to several different clusters or gaussian distributions with their respective probabilities.

To explain the probability of a point or points belonging to that certain cluster, we use something like EM (Expectation Maximization). This is the core algorithm that is used in the model and I will try to best explain what this algorithm does.

# Expectation-Maximization

In statistics, EM algorithm iteratively aims to find the local maximum likelhood or maximum a posteriori of parameters in statistical models in cases where the equation cannot be solved directly. Typically, if we know all the parameters, we would take the derivates of the likelihood function with respect to all the unknown values.
We could just digress from it by thinking, models like Gaussian Mixture have latent or hidden variables (z) that have influence on whether if a point belongs to a certain group or cluster. But as mentioned above, solving them is an obstacle. And EM algorithm helps us to thread over it. I will talk a bit about these latent variables further ahead.

EM consists of two steps in general. The E (Expectation) step and the M (Maximization) step.

## Expectation-step:

```Нөхцөлт математик дундаж``` i think???

In the initialization on the e-step, we assume that we don't know the parameter values. As the dimension increases in our gaussian distribution, the amount of parameters (theta) will increase responding it. The algorithm usually picks initial random guesses of these paramters. 

For one dimension feature and two gaussian distribution model, we have 5 parameters:


$$
\mu_1, \mu_2 = \text{Mean}
$$

$$
\sigma^2_1, \sigma^2_2  = \text{Variance}
$$

$$
\pi = \text{Weight}
$$  

Since we have only one dimension of data, we have a vector of variance, but it's possible to think a vector of 2 different 1x1 matrix of covariance.

### Likelihood function

$$
\mathcal{L}(\theta|x_i) = \prod_{i=1}^N f(x_i|\theta)
$$

We take the log-likelihood insead of the normal likelihood function for these reasons:

1. Caluclating the sum is computationally less demanding than product
2. Simplifies linear calculation
3. Prevents underflow

### Log-Likelihood function

$$
\ell(\theta|x_i) = \log\left(\prod_{i=1}^N f(x_i|\theta)\right) = \sum_{i=1}^N \log f(x_i|\theta)
$$

### Posterior probability

Here we use Bayes Theorem and substitute it with our parameters find the posterior probability of that point.

$$
\gamma_i = \frac{\pi_1 \cdot \phi_2(x_i|\mu_2,\sigma^2_2)}{\pi_2 \cdot \phi_1(x_i|\mu_1,\sigma^2_1) + \pi_1 \cdot \phi_2(x_i|\mu_2,\sigma^2_2)}
$$

The gamma value answers given what I observed (xi | x1, x2, ...xn) which component or cluster is responsible for generating this data.

Whereas:

$$
\gamma_j = 1 - \gamma_i
$$

Also, if you think about it, the $\gamma$ is our way of pondering about the latent variables. In another words, $\gamma$ is our probabilistic guess of this hidden variable.

## Maximization-step:
Here we do nothing but maximize the likelihood of the parameters using the fresh-ly estimated $\gamma$

$$
\pi = \sum_{i = 1}^N(1-\gamma_i)/N
$$

$$
\mu_1 = \frac{\sum_{i = 1}^N(1-\gamma_i)x_i}{\sum_{i = 1}^N(1-\gamma_i)}
$$

$$
\mu_2 = \frac{\sum_{i = 1}^N\gamma_ix_i}{\sum_{i = 1}^N\gamma_i}
$$


$$
\sigma_1 =
\sqrt{
\frac{\sum_{i = 1}^N(1-\gamma_i)(x_i - \mu_1)^2}{\sum_{i = 1}^N(1-\gamma_i)}
}
$$


$$
\sigma_2 =
\sqrt{
\frac{\sum_{i = 1}^N\gamma_i(x_i - \mu_2)^2}{\sum_{i = 1}^N\gamma_i}
}
$$

The algorithm iteratively runs this until convergence or maximum number of iterations reached. Remember, convergence meaning, the log-likelihood plateaus.

So, each point of data has a $\gamma$ value, meaning they have membership in each distribution.

# Multivariate Normal Distribution

We assume we have vector of more than one feature. 


$$
\Chi \sim \mathcal{N}(\Mu,\Sigma)
$$


$$
\begin{bmatrix}
X_1
\cr
X_2
\cr
\vdots
\cr
X_n
\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
\mu_{X_1}
\cr
\mu_{X_2}
\cr
\vdots
\cr
\mu_{X_n}
\end{bmatrix},
\begin{bmatrix}
\sigma_{X_1}^2 & \sigma_{X_1, X_2} & \dots & \sigma_{X_1, X_n}
\cr
\sigma_{X_2, X_1} & \sigma_{X_2}^2 & & \vdots
\cr
\vdots & & \ddots
\cr
\sigma_{X_n, X_1} & \ldots & & \sigma_{X_n}^2
\end{bmatrix}
\right)
$$


* The way we calculate normal distribution has changed a bit. We use multivariate normal distribution probability density function with a small tweak.


$$
f\mathrm{x}(x) = |2\pi\Sigma|^{-1/2}\exp\{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x - \mu)\}
$$

# Image Segmentation

We will be using Scikit-learn's Gaussian Mixture Model to segment images into different tones.

$$
\mu_i = {\mu_{R}, \mu_{G}, \mu_{B}}
$$

$$
\Sigma_i = 
\begin{bmatrix}
\sigma^2_{R} & \sigma_{RG} & \sigma_{RB}
\cr
\sigma_{RG} & \sigma^2_{G} & \sigma_{GB}
\cr
\sigma_{RB} & \sigma_{GB} & \sigma^2_{B}
\end{bmatrix}
$$

I think for this report we should try to minimize the different amount of regions in the image. Like set amount of object at the same time.



