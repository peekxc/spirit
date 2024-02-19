# %% 
from typing import *
import scipy as sp
import numpy as np 
import splex as sx
from scipy.spatial.distance import pdist 
from scipy.sparse.linalg import eigsh
from spirit.apparent_pairs import SpectralRI, deflate_sparse, UpLaplacian
from pbsig.persistence import sliding_window

# %% 
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

# %% Choose the function to optimize the slidding window
f = lambda t: np.cos(t) + np.cos(3*t)
# f = lambda t: np.cos(t)
SW = sliding_window(f, bounds=(0, 12*np.pi))

# %% Plot periodic function
dom = np.linspace(0, 12*np.pi, 1200)
p = figure(width=450, height=225)
p.line(dom, f(dom))
show(p)

# %% Make a slidding window embedding
from pbsig.linalg import pca
from pbsig.color import bin_color
N, M = 120, 24 # num of points, dimension 
X_delay = SW(n=N, d=M, L=6) ## should be the perfect period

pt_color = (bin_color(np.arange(len(X_delay)), "turbo")*255).astype(np.uint8)
p = figure(width=250, height=250, match_aspect=True)
p.scatter(*pca(X_delay).T, color=pt_color)
show(p)

# %% Use ripser to infer range of [b, d]
from ripser import ripser
from persim import plot_diagrams
diagrams = ripser(SW(n=N, d=M, L=6))['dgms']
lifetimes = np.diff(diagrams[1], axis=1).flatten()
h1_bd = diagrams[1][np.argmax(lifetimes),:]
print(f"Most persistent H1 cycle: { h1_bd }")
birth, death = h1_bd[0]*1.20, h1_bd[1]*0.20
plot_diagrams(diagrams, show=False)

# %% Evaluate spectral rank invariant on full complex 
from itertools import combinations
S = sx.RankComplex([[i] for i in np.arange(N)])
S.add(np.array(list(combinations(np.ravel(sx.faces(S,0)), 2))))
S.add(np.array(list(combinations(np.ravel(sx.faces(S,0)), 3))))

from spirit.apparent_pairs import SpectralRI, deflate_sparse, UpLaplacian
SI = SpectralRI(S)
X = SW(n=N, d=M, L=6)
dX = pdist(X)

## Start with an embedding + diameter filter 
diam_f = sx.flag_filter(dX)
SI._weights[0] = np.repeat(1e-8, sx.card(S,0))
SI._weights[1] = diam_f(sx.faces(S, 1))
SI._weights[2] = diam_f(sx.faces(S, 2))
SI.detect_pivots(dX, p=1, f_type="flag")
SI.detect_pivots(dX, p=2, f_type="flag")

# %%
a,b,c,d = h1_bd[0]*0.80, h1_bd[0]*1.20, h1_bd[1]*0.80, h1_bd[1]*1.20
SI.query(2,a,b,c,d, summands=True, method="cholesky")
# 1099 - 1103 - 5857 + 5862

# %% 
import timeit
timeit.timeit(lambda: ripser(X), number=10)
timeit.timeit(lambda: SI.query(2,a,b,c,d, summands=True, method="cholesky"), number=10)

from line_profiler import LineProfiler
profile = LineProfiler()
profile.add_function(SI.query)
profile.add_function(SI.rank)
profile.add_function(SI.lower_left)
profile.add_function(deflate_sparse)
profile.enable_by_count()
SI.query(2,a,b,c,d, summands=True, method="cholesky")
profile.print_stats()



# %% 
defl, ap = True, True
A1 = SI.lower_left(b,c,2, deflate=defl, apparent=ap).D
A2 = SI.lower_left(a,c,2, deflate=defl, apparent=ap).D
A3 = SI.lower_left(b,d,2, deflate=defl, apparent=ap).D
A4 = SI.lower_left(a,d,2, deflate=defl, apparent=ap).D

A1_lil = A1.tolil()
A1_defl = A1_lil[:,SI._status[2] <= 0]
r_base = np.linalg.matrix_rank((A1 @ A1.T).todense())
r_rm_ap = np.linalg.matrix_rank((A1_defl @ A1_defl.T).todense()) 


from sksparse.cholmod import cholesky, cholesky_AAt
from scipy.sparse import csc_matrix
import timeit
A2_csc = csc_matrix(A2)
beta = 1e-6
threshold = max(np.max(F) * max(A2_csc.shape) *  np.finfo(np.float32).eps, beta * 100)
timeit.timeit(lambda: np.sum(cholesky_AAt(A2_csc,  beta=beta).D() >= threshold), number=10)


np.linalg.matrix_rank((A1 @ A1.T).todense())
np.linalg.matrix_rank((A2_csc @ A2_csc.T).todense())
np.linalg.matrix_rank((A3 @ A3.T).todense())
np.linalg.matrix_rank((A4 @ A4.T).todense())
# 1099 - 1103 - 5857 + 5862
# 37 - 52 - 312 + 328
list(SI.query(a,b,c,d))

0 - 69 - 104 + 173

hutch(A, fun="log", deg=10, orth=5, ncv=7, maxiter=3, verbose=True, info=True)


# %% Enumerate the Persisent Betti numbers
from pbsig.persistence import sw_parameters
d,tau_min = sw_parameters(bounds=(0, 12*np.pi), d=M, L=24)
d,tau_max = sw_parameters(bounds=(0, 12*np.pi), d=M, L=1)

import splex as sx
from itertools import combinations
from primate.trace import hutch
from primate.functional import numrank 

query = BettiQuery(S, p = 1)
query.q_solver = lambda L: 0 if np.sum(L.shape) == 0 else numrank(L, atol=0.50, gap="simple")
query.p_solver = lambda L: 0 if np.sum(L.shape) == 0 else numrank(L, atol=0.50, gap="simple")

# %% Parameterize the filtration
from scipy.spatial.distance import pdist
X = F(n=N, d=M, L=6)
diam_f = sx.flag_filter(pdist(X))
query.weights[0] = np.ones(sx.card(S, 0))
query.weights[1] = diam_f(sx.faces(S, 1)) 
query.weights[2] = diam_f(sx.faces(S, 2)) 

# %% Show ripser 
from ripser import ripser
from pbsig.vis import figure_dgm
dgm_dtype = [('birth', 'f4'), ('death', 'f4')]
dgm_h1 = ripser(X)['dgms'][1]
dgm_h1 = np.array([tuple(p) for p in dgm_h1], dtype=dgm_dtype)
show(figure_dgm(dgm_h1))

# %% TODO: apparent pairs optimization, new Weighted Laplacian operator, better spectral gap estimation
query.q_solver = lambda L: 0 if np.sum(L.shape) == 0 else numrank(L, atol=0.50, gap="auto")
query.p_solver = lambda L: 0 if np.sum(L.shape) == 0 else numrank(L, atol=0.50, gap="auto")
a,b = 2, 6
query(i=a, j=b, terms=True)


np.sum(query.operator(0, i=a, j=b).data > 0)
np.linalg.matrix_rank(query.operator(1, i=a, j=b, deflate=True).todense())
np.linalg.matrix_rank(query.operator(2, i=a, j=b, deflate=True).todense())
np.linalg.matrix_rank(query.operator(3, i=a, j=b, deflate=True).todense())

A = query.operator(1, i=a, j=b, deflate=True)

# numrank(A, atol = 0.10, maxiter=1500, gap="simple")

# from primate.trace import hutchpp
# hutchpp(A)

# from pbsig.apparent_pairs import apparent_pairs
# apparent_pairs(S, f=diam_f, p = 1)

# p = figure(width=250, height=250, match_aspect=True)
# p.scatter(*pca(F(n=N, d=M, tau=0.5*tau_max)).T)
# show(p)

# PBS = np.array([persistent_betti_rips(F(n=N, d=M, tau=tau), b=birth, d=death, summands=True) for tau in T])
# PB = PBS[:,0] - (PBS[:,1] + PBS[:,2])
