testrembed.m
==========
This file exhibits the basic guarantee of the randomized embedding code.

Given features \\( X \\) and labels \\( Y \\), where the SVD of X is given by \\( X = U_X \Sigma_X V_X^\top \\), and \\( U_X^\top Y = U_E \Sigma_E V_E \\), the embedding is defined as the first \\(k\\) columns of \\( V_E \\).  This definition is motivated by the optimal rank-constrained least-squares approximation of \\( Y \\) given \\( X \\), as explained in [this paper](http://arxiv.org/abs/1412.6547).

Randomized methods provide a fast way of approximating these SVDs when the
dimensionalities are large.
