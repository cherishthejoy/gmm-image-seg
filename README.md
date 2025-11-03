# gmm-image-seg
Gaussian mixture model image segmentation

## Likelihood function

$$
\mathcal{L}(\theta|x_i) = \prod_{i=1}^n f(x_i|\theta)
$$

## Log-Likelihood function

$$
\ell(\theta|x_i) = \log\left(\prod_{i=1}^n f(x_i|\theta)\right) = \sum_{i=1}^n \log f(x_i|\theta)
$$

## Posterior probability

$$
\gamma_i = \frac{\pi_1 \cdot \phi_2(x_i|\mu_2,\sigma^2_2)}{\pi_2 \cdot \phi_1(x_i|\mu_1,\sigma^2_1) + \pi_1 \cdot \phi_2(x_i|\mu_2,\sigma^2_2)}
$$

Whereas:

$$
\gamma_j = 1 - \gamma_i
$$