# COMP579 Final Project
Topic: Accelerating training in deep reinforcement learning using predictive variance reduction

Various reinforcement learning (RL) algorithms suffer from the instability brought by large variance, a by-product of approximation processes including value function approximation, bootstrapping or off-policy learning (if the method is offline). The case where all three factors are combined is described by the term deadly triad, which cause the algorithm to converge to an unstable, sub-optimal policies.

Our primary focus in this project is to tackle the challenge associated with the first factor – value function approximation. This approximation is often done using stochastic gradient descend (SGD), which is a popular method for optimization but inherently has large variance. Our solution to remedy this is through known variance reduction techniques such as stochastic variance reduce gradient (SVRG)1 or stochastic average gradient (SAG)2. We would like to adapt these techniques for several RL algorithms such as SAC or A3C, and investigate whether this yields an advantage on the convergence speed and stability of the algorithm. In addition, other variance reduction techniques including variance reduced extragradient3 from generative adversarial network (GAN) or spectral normalization of layers (SNAC)4 could be incorporated and tested.

Our contributions can be summarized as follows:
1.	We extensively evaluate the effectiveness of predictive variance reduction on decreasing the variance of gradients.
2.	We empirically test whether low gradient variances could lead to a better stability of RL algorithms, and how significant the advantage is.
3.	We present benchmarking results and ablation studies on complex control problems, and compare the performances to state-of-the-art algorithms including SAC and A3C.
