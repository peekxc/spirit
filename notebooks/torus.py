# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: spri
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from comb_laplacian.sampler import sample_torus_tube
from landmark import landmarks
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.layouts import row, column
output_notebook(hide_banner=True, verbose=False)

# %%
np.random.seed(1234)
X = sample_torus_tube(1500, ar=3.0, seed=1234)
X = X[landmarks(X, 50)]


# %%
p = figure(width=300, height=300)
q = figure(width=300, height=300)
p.scatter(*X[:,[0,1]].T)
q.scatter(*X[:,[0,2]].T)
show(row(p, q))

# %%
from ripser import ripser
from pbsig.vis import figure_dgm
dgms = ripser(X, maxdim=1)['dgms']
show(figure_dgm(dgms[0], title="H0"))
show(figure_dgm(dgms[1], title="H1"))


# %% 
from spirit import SpectralRI
RI = SpectralRI(n=len(X), max_dim=1)
RI.construct_flag(X, max_dim=1, apparent=True, discard_pos=True)
RI._simplices[1]
RI.construct_operator(0, form="boundary")
RI.construct_operator(1, form="boundary")
# RI.construct_operator(2, form="boundary")
RI._ops
# RI.construct_operator(1, form="boundary")

# %% 
RI.query_dim(p=0, a=0.1, b=0.5, summands=False)
RI.query_dim(p=1, a=0.6, b=1.2, summands=False)
RI.lower_left(p=0, i=-np.inf, j=0.6)
# RI.lower_left(p=1, i=-np.inf, j=0.6)

# %% 
# RI.lower_left(p=1, i=-np.inf, j=0.6)
p = 1
q = p + 1
i, j = -np.inf, 0.6
p_inc = np.logical_and(RI._weights[p] >= i, RI._weights[p] <= j)
q_inc = np.logical_and(RI._weights[q] >= i, RI._weights[q] <= j)
S_ll = RI._simplices[q][q_inc]
F_ll = RI._simplices[p][p_inc]



rank_to_comb(19067, k=3, order='colex', n=len(X))

comb_to_rank([[13,36],[13,49],[36,49]], k=2, order='colex', n=len(X))

643 in F_ll

from combin import rank_to_comb
rank_to_comb(S_ll, k=3, n=len(X), order='colex')



from comb_laplacian.laplacian_cpu import k_boundary_cpu
n,k = len(X), 3
BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(6)]).astype(np.int64)
k_faces = np.zeros(k, dtype=np.int64)
all_faces = []
for s in S_ll:
  k_boundary_cpu(simplex=s, dim=2, n=RI.n, BT=BT, out=k_faces)
  all_faces.extend(k_faces.copy())

RI._simplices[1][1058]
RI._weights[1][1058]
RI._simplices[1]
np.logical_and(self._weights[p] >= i, self._weights[p] <= j)[1058]

166 in RI._simplices[p][p_inc]

rank_to_comb(166, order='colex', n=len(X), k=2)
np.unique(all_faces)
np.unique(RI._simplices[1][RI._weights[1] <= 0.6])
np.unique(RI._simplices[2][RI._weights[2] <= 0.6])

rank_to_comb(F_ll, k=2, n=len(X), order='colex')

np.max(RI._simplices[1]) < comb(len(X), 2)
np.max(RI._simplices[2]) < comb(len(X), 3)

# %% 


# from spirit.apparent_pairs import UpLaplacian, boundary_matrix
# D2 = boundary_matrix(p=2, p_simplices=RI._simplices[2], f_simplices=RI._simplices[1])

# L = UpLaplacian(D2)
# L.lower_left(i=0.6, j=0.8)

# RI.query_dim(p=1, a=0.6, b=0.8)
