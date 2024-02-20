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

pt_color = (bin_color(np.arange(N), "turbo")*255).astype(np.uint8)
p = figure(width=250, height=250, match_aspect=True)
p.scatter(*pca(X_delay, center=True).T, color=pt_color)
show(p)

# %% Use ripser to infer range of [b, d]
from ripser import ripser
from persim import plot_diagrams
diagrams = ripser(SW(n=N, d=M, L=6))['dgms']
plot_diagrams(diagrams, show=False)

# %% Evaluate spectral rank invariant on full complex 
from itertools import combinations
X = SW(n=N, d=M, L=6)
S = sx.rips_complex(X, p=2)

from line_profiler import LineProfiler
profile = LineProfiler()
profile.add_function(SpectralRI)
profile.add_function(SpectralRI.__init__)
profile.enable_by_count()
SI = SpectralRI(S)
profile.print_stats()


profile = LineProfiler()
profile.add_function(SI.detect_pivots)
profile.enable_by_count()
SI.detect_pivots(dX, p=2, f_type="flag")
profile.print_stats()

# %% 
from spirit.apparent_pairs import SpectralRI, deflate_sparse, UpLaplacian
SI = SpectralRI(S)
dX = pdist(X)

## Start with an embedding + diameter filter 
from combin import rank_to_comb
diam_f = sx.flag_filter(dX)
SI._weights[0] = np.repeat(1e-8, sx.card(S,0))
SI._weights[1] = diam_f(rank_to_comb(SI.simplices[1], k=2, order='colex', n=sx.card(S,0)))
SI._weights[2] = diam_f(rank_to_comb(SI.simplices[2], k=3, order='colex', n=sx.card(S,0)))
SI.detect_pivots(dX, p=1, f_type="flag")
SI.detect_pivots(dX, p=2, f_type="flag")

# %%
# a,b,c,d = h1_bd[0]*0.80, h1_bd[0]*1.20, h1_bd[1]*0.80, h1_bd[1]*1.20 # 1099 - 1103 - 5857 + 5862
a,b,c,d = 0.50, 2.0, 6.0, 8.0
SI.query(1,a,b,c,d, summands=True, method="cholesky")
# 977, 1225, 5369, 5618


# %% Vary the step size 
from pbsig.persistence import sw_parameters
_, tau_ub = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=2)
_, tau_lb = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=12)
M_opt, tau_opt = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=6)

for tau in np.linspace(tau_lb, tau_ub, 10):
  X = SW(n=N, d=M, tau=tau)
  dX = pdist(X)
  diam_f = sx.flag_filter(dX)
  SI._weights[0] = np.repeat(1e-8, sx.card(S,0))
  SI._weights[1] = diam_f(rank_to_comb(SI.simplices[1], k=2, order='colex', n=sx.card(S,0)))
  SI._weights[2] = diam_f(rank_to_comb(SI.simplices[2], k=3, order='colex', n=sx.card(S,0)))
  SI._status[2].fill(0)
  SI.detect_pivots(dX, p=2, f_type="flag")
  print(f"Step size: {tau:.4f}, Multiplicity: {SI.query(1,a,b,c,d, summands=False, method='cholesky')}")


# timeit.timeit(lambda: SI.query(1,b,c,summands=True, method='cholesky'), number=10)
# timeit.timeit(lambda: SI.query(1,a,b,c,d,summands=True, method='cholesky'), number=10)

# %% 
from pbsig.vis import figure_dgm, show, Range1d
diagrams = ripser(SW(n=N, d=M, tau=0.6716))['dgms']
p = figure_dgm(diagrams[1])
p.rect(x=a + (b-a)/2, y=c + (d-c)/2, width=b-a,height=d-c, fill_alpha=0)
p.y_range = Range1d(0, 9)
p.x_range = Range1d(0, 9)
show(p)


# %% 
import timeit
timeit.timeit(lambda: ripser(X), number=10)
timeit.timeit(lambda: SI.query(1,a,b,c,d, summands=True, method="cholesky"), number=10)
timeit.timeit(lambda: SI.query(1,b,c, summands=True, method="cholesky"), number=10)
timeit.timeit(lambda: SI.detect_pivots(dX, p=2, f_type="flag"), number=1)


from spirit.apparent_pairs import compress_index, deflate_sparse
from line_profiler import LineProfiler
profile = LineProfiler()
profile.add_function(SI.query)
profile.add_function(SI.rank)
profile.add_function(SI.lower_left)
profile.add_function(deflate_sparse)
profile.add_function(compress_index)
profile.enable_by_count()
SI.query(2,a,b,c,d, summands=True, method="cholesky")
profile.print_stats()

from line_profiler import LineProfiler
profile = LineProfiler()
profile.add_function(SI.detect_pivots)
profile.enable_by_count()
SI.detect_pivots(dX, p=2, f_type="flag")
profile.print_stats()

# %% Test coo deflation 
## Expand = False is about 10% faster, should use less memory
import timeit
timeit.timeit(lambda: SI.lower_left(b,c, p = 2, deflate=True, apparent=True, expand=True), number=200)
timeit.timeit(lambda: SI.lower_left(b,c, p = 2, deflate=True, apparent=True, expand=False), number=200)



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
