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

# %% Imports
import numpy as np
from comb_laplacian.sampler import sample_torus_tube
from landmark import landmarks
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.layouts import row, column

output_notebook(hide_banner=True, verbose=False)

# %% Sample torus
np.random.seed(1234)
X = sample_torus_tube(1500, ar=3.0, seed=1234)
X = X[landmarks(X, 150)]

# %% Plot cross sections
p = figure(width=300, height=300)
q = figure(width=300, height=300)
p.scatter(*X[:, [0, 1]].T)
q.scatter(*X[:, [0, 2]].T)
show(row(p, q))

# %% Plot persistence diagrams
from ripser import ripser
from pbsig.vis import figure_dgm

dgms = ripser(X, maxdim=1)["dgms"]
# show(figure_dgm(dgms[0], title="H0"))
show(figure_dgm(dgms[1], title="H1"))

# %% Construct Laplacian operators
from spirit import SpectralRI

RI = SpectralRI(n=len(X), max_dim=1)
RI.construct_flag(X, max_dim=1, apparent=True, discard_pos=True)
RI.construct_operator(0, form="boundary")
RI.construct_operator(1, form="boundary")

# %% Query dimension
RI.query_dim(p=1, a=0.6, b=1.2, summands=False, method="cholesky") == 1
RI.query_dim(p=1, a=0.2, b=0.4, c=0.5, d=1.3, summands=False, method="cholesky") == 2

# %% Figure out how to estimate trace
RI.query_dim(p=1, a=0.6, b=1.2, summands=True, method="trace", maxiter=100, deg=200)

RI.query_dim(p=1, a=0.6, b=1.2, summands=True, method="cholesky")
# 1044, 149, 3951, 3057

L_prob = RI.lower_left(1, -np.inf, 1.2)  # rank = 3951

# from primate.functional import numrank
# from primate.trace import hutch
# hutch(L_prob, deg=200, fun="numrank", threshold=1e-6, verbose=True, maxiter=100)
# numrank(L_prob, gap=1e-6, deg=200, maxiter=100, verbose=True)

summands = RI.query_dim(p=1, a=0.6, b=1.2, summands=True, method="trace", maxiter=100, deg=300, verbose=True)

t1 = 1044
t2 = np.round(148.307 + 2.253)
t3 = np.round(3956.664 + 12.397)
t4 = np.round(3057.000 - 0.000)
min_bnd = max(t1 - t2 - t3 + t4, 0.0)
max_bnd = 1044 - (149.115 - 0.428) - (3948.978 - 3.503) + t4


min_bnd = 1044 - 148

# %% Find pairs in [a,b]x[c,d]
H1_pairs = RI.query_pairs(p=1, a=0.2, b=0.4, c=0.5, d=1.3)
show(figure_dgm(H1_pairs, title="H1"))

# %% Find
# from spirit.apparent_pairs import subset_boundary
# p_inc = np.logical_and(RI._weights[1] >= 0.2, RI._weights[1] <= 0.5)
# q_inc = np.logical_and(RI._weights[2] >= 0.2, RI._weights[2] <= 0.5)
# D = subset_boundary(RI._ops[1].D, p_inc, q_inc)
# np.linalg.matrix_rank(D[0].todense())
# from scipy.sparse import save_npz
# save_npz("/Users/mpiekenbrock/spirit/data/D2_665_rank512.npz", matrix=D[0], compressed=True)


# timeit.timeit(lambda: RI._ops[1] @ np.random.uniform(size=11175, low=0, high=1), number = 11175)

# %%
RI._simplices[1]
RI.construct_operator(0, form="matrix free")
RI.construct_operator(1, form="matrix free")

RI._ops
RI.construct_operator(0, form="boundary")
RI.construct_operator(1, form="boundary")


# %%
import timeit

RI.query_dim(p=0, a=0.1, b=0.5, summands=True, method="trace", verbose=True, deg=50)

RI.construct_operator(0, form="boundary")
RI.construct_operator(1, form="boundary")
RI.query_dim(p=1, a=0.6, b=1.2, summands=True, method="cholesky")

RI.query_pairs(p=1, a=0.2, b=0.4, c=1.0, d=1.4)


RI.query_dim(p=1, a=0.6, b=1.2, summands=True, method="trace", verbose=True, deg=50)

op = RI.lower_left(p=1, i=-np.inf, j=1.2)

from spirit.apparent_pairs import to_canonical
from scipy.sparse.linalg import eigsh

# eigsh(op, which="LM", k=1, return_eigenvectors=False)
# 0.5/16
# np.linalg.matrix_rank(op.tosparse().todense())
true_ew = np.linalg.eigh(op.tosparse().todense())[0]

(sd, bins), info = spectral_density(op, fun="smoothstep", a=1e-6, b=0.047, deg=40, bins=1500)
mean_confidence_interval(info["quad_est"])

from scipy.sparse import save_npz

save_npz("UpLaplacian1_446_rank396.npz", to_canonical(op.tosparse()))


# numrank(op)

# %% Profile sparse matrix
op = RI.lower_left(p=1, i=-np.inf, j=1.2)
x = np.random.uniform(size=op.shape[1])
timeit.timeit(lambda: op @ x, number=20 * 150)

op_D = op.D.tocsc()
op_Dt = op_D.T.tocsc()
op_D.sort_indices()
op_Dt.sort_indices()
x = np.random.uniform(size=op_D.shape[1])
timeit.timeit(lambda: op_D @ x, number=20 * 150)
timeit.timeit(lambda: op_D.T @ (op_D @ x), number=20 * 150)
timeit.timeit(lambda: op_Dt @ (op_D @ x), number=20 * 150)

RI.construct_operator(0, form="matrix free")
RI.construct_operator(1, form="matrix free")
RI.query_dim(p=1, a=0.6, b=1.2, summands=True, method="trace", verbose=True, deg=50, maxiter=50)

op = RI.lower_left(p=1, i=-np.inf, j=1.2)
numrank(op, verbose=True)

from primate.functional import spectral_density, estimate_spectral_radius

spectral_density(op, fun="smoothstep", a=1e-6, b=0.01)

from line_profiler import LineProfiler

profiler = LineProfiler()
profiler.add_function(spectral_density)
profiler.enable_by_count()
spectral_density(op, fun="smoothstep", a=1e-6, b=0.01)
profiler.print_stats(output_unit=1e-6)

import timeit

timeit.timeit(lambda: op @ np.random.uniform(size=446), number=20 * 150)
timeit.timeit(lambda: sl_gauss(op, deg=20, n=150), number=1)

from primate.quadrature import sl_gauss
from primate.functional import numrank

numrank(op)

RI.lower_left(p=0, i=-np.inf, j=0.6)
# RI.lower_left(p=1, i=-np.inf, j=0.6)

RI.construct_operator(0, form="boundary")
RI.construct_operator(1, form="boundary")
RI.query_dim(p=1, a=0.6, b=1.2, summands=False, method="cholesky")

# %% Verification idea
## To verify: 281 = 199 + 82, c = a + b
A = lambda: np.random.normal(loc=199, scale=30)
B = lambda: np.random.normal(loc=82, scale=30)
C = lambda: np.random.normal(loc=281, scale=30)
a, b, c = A(), B(), C()
i = 0
while int(c) != int(a + b):
	a, b, c = A(), B(), C()
	i += 1

import random

unit = np.array([-1, 1])
rc = 281 + unit * np.random.choice(range(3), size=1)  # (275, 291)
ra = 199 + unit * np.random.choice(range(3), size=1)
rb = 82 + unit * np.random.choice(range(3), size=1)  # (180, 212)

A = lambda: random.randint(*ra)
B = lambda: random.randint(*rb)
C = lambda: random.randint(*rc)
a, b, c = A(), B(), C()
i = 0
while c != a + b:
	a, b, c = A(), B(), C()
	jj = random.randint(0, 2)
	if jj == 0:
		a = A()
	elif jj == 1:
		b = B()
	else:
		c = C()
	i += 1
print(i)


# %%
# RI.lower_left(p=1, i=-np.inf, j=0.6)
p = 1
q = p + 1
i, j = -np.inf, 0.6
p_inc = np.logical_and(RI._weights[p] >= i, RI._weights[p] <= j)
q_inc = np.logical_and(RI._weights[q] >= i, RI._weights[q] <= j)
S_ll = RI._simplices[q][q_inc]
F_ll = RI._simplices[p][p_inc]


rank_to_comb(19067, k=3, order="colex", n=len(X))

comb_to_rank([[13, 36], [13, 49], [36, 49]], k=2, order="colex", n=len(X))

643 in F_ll

from combin import rank_to_comb

rank_to_comb(S_ll, k=3, n=len(X), order="colex")


from comb_laplacian.laplacian_cpu import k_boundary_cpu

n, k = len(X), 3
BT = np.array([[int(comb(ni, ki)) for ni in range(n + 1)] for ki in range(6)]).astype(np.int64)
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

rank_to_comb(166, order="colex", n=len(X), k=2)
np.unique(all_faces)
np.unique(RI._simplices[1][RI._weights[1] <= 0.6])
np.unique(RI._simplices[2][RI._weights[2] <= 0.6])

rank_to_comb(F_ll, k=2, n=len(X), order="colex")

np.max(RI._simplices[1]) < comb(len(X), 2)
np.max(RI._simplices[2]) < comb(len(X), 3)

# %%


# from spirit.apparent_pairs import UpLaplacian, boundary_matrix
# D2 = boundary_matrix(p=2, p_simplices=RI._simplices[2], f_simplices=RI._simplices[1])

# L = UpLaplacian(D2)
# L.lower_left(i=0.6, j=0.8)

# RI.query_dim(p=1, a=0.6, b=0.8)
