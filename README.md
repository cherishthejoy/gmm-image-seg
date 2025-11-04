# gmm-image-seg

Gaussian Mixture Model is a "probabilistic" model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with "unknown" parameters.

Gaussian Mixture Model utilizes the idea of "soft-assignment" opposed to k-means clustering "hard-assignment" where each data point could belong to several different clusters or gaussian distributions with their respective probabilities.

To explain the probability of a point or points belonging to that certain cluster, we use something like EM (Expectation Maximization). This is the core algorithm that is used in the model and I will try to best explain what this algorithm does.

# Expectation-Maximization
Нөхцөлт дундаж

In statistics, EM algorithm iteratively aims to find the local maximum likelhood or maximum a posteriori of parameters in statistical models in cases where the equation cannot be solved directly. Typically, if we know all the parameters, we would take the derivates of the likelihood function with respect to all the unknown values.
We could just digress from it by thinking, models like Gaussian Mixture have latent or hidden variables (z) that have influence on whether if a point belongs to a certain group or cluster. But as mentioned above, solving them is an obstacle. And EM algorithm helps us to thread over it. I will talk a bit about these latent variables further ahead.

EM consists of two steps in general. The E (Expectation) step and the M (Maximization) step.

## Expectation-step:

In the initialization on the e-step, we assume that we don't know the parameter values. As the dimension increases in our gaussian distribution, the amount of parameters (theta) will increase responding it. The algorithm usually picks initial random guesses of these paramters. 

For one dimension feature and two gaussian distribution model, we have 5 parameters:

$$
\mu_1, \mu_2 = Mean
$$

$$
\sigma^2_1, \sigma^2_2 = Variance
$$

$$
\pi = Weight
$$ 



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

So if each point of data has a $\gamma$ value, meaning they have membership in each distribution.