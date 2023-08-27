# -*- coding: utf-8 -*-
# Dimensional examples (using 3 maps, each with 5 cities as an example)
# Encoder(4-dimensional)                          Decoder(5-dimensional)
# Q             [8, 3, 5, 16]           glimpse_Q       [8, 3, 1, 1, 16]
# K             [8, 3, 5, 16]           glimpse_K       [8, 3, 1, 5, 16]
# V             [8, 3, 5, 16]           glimpse_V       [8, 3, 1, 5, 16]
# compatibility [8, 3, 5, 5]            compatibility   [8, 3, 1, 1, 5]
# Encoder: (n_heads, batch_size, graph_size, key/val_size)
# Decoder: (n_heads, batch_size, num_steps, graph_size, key_size)
# Reminder: When programming, try to handle using the last two dimensions (i.e., -1, -2) to avoid errors
import math
import torch
import torch.nn as nn


# Called by encoder-decoder，mainly to determine which to use when calculating compatibility
# EDtype : setting encoder & decoder                    (Available values: encoder, decoder)
# ctype  : Used to specify compatibility type (can be omitted for now) (Available values: encoder, decoder)
def compatibility_factory(n_heads, embed_dim, EDtype=None, ctype=None):
    #return Scaled_Dot_Product(n_heads,embed_dim,EDtype)
    return Cauchy(n_heads,embed_dim,EDtype)


# Parent class for kernel methods, mainly to pull in parameters and set some common functions for subsequent processing
class Compatibility(nn.Module):
    def __init__(self, n_heads, embed_dim, EDtype=None):
        super(Compatibility, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dim = self.embed_dim // self.n_heads
        self.EDtype = EDtype
        self.pCnt = 0

    # normalize method, which maps data to [a,b], the formula is referenced from wiki
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    def normalize_compatibility(self, compatibility, a=-9, b=9):
        max_vals = torch.max(torch.max(compatibility, dim=-1)[0], dim=-1)[0]
        min_vals = torch.min(torch.min(compatibility, dim=-1)[0], dim=-1)[0]
        return a + ((compatibility - min_vals[..., None, None]) * (b - a) / (max_vals - min_vals)[..., None, None])
        #max = torch.amax(compatibility, dim=(-2,-1), keepdim=True)
        #min = torch.amin(compatibility, dim=(-2,-1), keepdim=True)
        #compatibility = a + ((compatibility-min)*(b-a) / (max-min))
        #return compatibility

    # point-wise Euclidean distance (weight matrix)
    def euclidean_matrix(self, Q, K):
        return torch.sqrt(
            torch.sum(Q ** 2, dim=-1).unsqueeze(-1)
            +torch.sum(K ** 2, dim=-1).unsqueeze(-2)
            -2*torch.matmul(Q, K.transpose(-2, -1))
        )
            

    def forward(self, Q, K):
        #if self.EDtype == 'encoder':
        if len(Q.shape) == 4:
            n_heads, batch_size, n_query, dim = Q.shape
            n_heads, batch_size, graph_size, dim = K.shape
        else:
            n_heads, batch_size, num_steps, n_query, dim = Q.shape
            n_heads, batch_size, num_steps, graph_size, dim = K.shape
            self.num_steps = num_steps
        self.batch_size = batch_size
        self.n_query = n_query
        self.graph_size = graph_size
        self.dim = dim


# Matrix Multiplication (Scaled Dot Product)
class Scaled_Dot_Product(Compatibility):
    def __init__(self, n_heads, embed_dim, EDtype=None):
        super(Scaled_Dot_Product, self).__init__(n_heads, embed_dim, EDtype)
        self.norm_factor = 1 / math.sqrt(self.dim)

    def forward(self, Q, K):
        super().forward(Q, K)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(-2, -1))
        return compatibility


# Cauchy kernel
# K(x, y) = 1 / (1 + (||x - y||/sigma)^ 2)
class Cauchy(Compatibility):
    def __init__(self, n_heads, embed_dim, EDtype=None, sigma=1.0):
        super(Cauchy, self).__init__(n_heads, embed_dim, EDtype)
        #self.sigma = nn.Parameter(torch.rand(1))
        self.sigma = sigma

    def forward(self, Q, K):
        super().forward(Q, K)
        # compute ||x - y||
        dist = self.euclidean_matrix(Q,K)
        # compute  1 / (1 + (||x - y||/sigma)^ 2)
        compatibility = 1 / (1 + (dist/self.sigma).pow(2))
        return compatibility


# Gaussian Radial Basis Function Kernel
# K(x, y) = e^(  (-1*||x - y||^2) / (2*sigma^2)  )
class RBFKernel_L2(Compatibility):
    def __init__(self, n_heads, embed_dim, EDtype=None, RBF_gamma=0.025):
        super(RBFKernel_L2, self).__init__(n_heads, embed_dim, EDtype)
        self.RBF_gamma = RBF_gamma

    def forward(self, Q, K):
        super().forward(Q, K)
        # compute ||x - y||
        dist = self.euclidean_matrix(Q,K)
        compatibility = torch.exp(-1*self.RBF_gamma*dist.pow(2))
        #compatibility = torch.exp((-1*dist.pow(2))/(2*self.RBF_sigma**2))
        return compatibility


# Polynomial Kernel
# K(x, y) = ( alpha*<x, y> +c)^d
# Parameters reference https://journals.sagepub.com/doi/pdf/10.1260/1748-3018.8.2.163
class Polynomial_Kernel(Compatibility):
    def __init__(self, n_heads, embed_dim, EDtype=None):
        super(Polynomial_Kernel, self).__init__(n_heads, embed_dim, EDtype)
        self.poly_alpha = 1.0
        self.poly_c = 1.0
        self.poly_d = 1.0

    def forward(self, Q, K):
        super().forward(Q, K)
        compatibility = (self.poly_alpha * torch.matmul(Q, K.transpose(-2, -1)) + self.poly_c).pow(self.poly_d)
        return compatibility



