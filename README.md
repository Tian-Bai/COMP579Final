# COMP579 Final Project
Topic: Accelerating training in deep reinforcement learning using predictive variance reduction

Various reinforcement learning (RL) algorithms suffer from the instability brought by large variance, a by-product of approximation processes including value function approximation, bootstrapping or off-policy learning (if the method is off-policy). The case where all three factors are combined is described by the term deadly triad, which cause the algorithm to converge to an unstable, sub-optimal policies.

Our primary focus in this project is to tackle the challenge associated with the first factor – value function approximation. This approximation is often done using stochastic gradient descend (SGD), which is a popular method for optimization but inherently has large variance. Our solution to remedy this is through known variance reduction techniques such as stochastic variance reduce gradient (SVRG) or stochastic average gradient (SAG). We would like to adapt these techniques for several RL algorithms such as SAC or A3C, and investigate whether this yields an advantage on the convergence speed and stability of the algorithm. In addition, other variance reduction techniques including variance reduced extragradient from generative adversarial network (GAN) or spectral normalization of layers (SNAC) could be incorporated and tested.

Our contributions can be summarized as follows:
1.	We extensively evaluate the effectiveness of predictive variance reduction on decreasing the variance of gradients.
2.	We empirically test whether low gradient variances could lead to a better stability of RL algorithms, and how significant the advantage is.
3.	We present benchmarking results and ablation studies on complex control problems, and compare the performances to state-of-the-art algorithms including SAC and A3C.

References:
1. Johnson, R., Zhang, T.: Accelerating stochastic gradient descent using predictive variance reduction. Adv. Neural Inf. Process. Syst. 26, 315–323 (2013).
2. Nicolas Le Roux, Mark Schmidt, and Francis Bach. A Stochastic Gradient Method with an Exponential Convergence Rate for Strongly-Convex Optimization with Finite Training Sets. arXiv preprint arXiv:1202.6258, 2012.
3. Tatjana C., Gauthier G., François F. and Simon L. Reducing Noise in GAN Training with Variance Reduced Extragradient. arXiv preprint arXiv: 1904.0859, 2020.
4. Payal Bawa, Rafael Oliveira, and Fabio Ramos. Variance Reduction in Off-Policy Deep Reinforcement Learning using Spectral Normalization. Deep Reinforcement Learning Workshop NeurIPS 2022, 2022.
