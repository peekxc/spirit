# %% 
from typing import *
import scipy as sp
import numpy as np 
import splex as sx
from scipy.spatial.distance import pdist 
from scipy.sparse.linalg import eigsh
# from spirit.apparent_pairs import SpectralRI, deflate_sparse, UpLaplacian
from pbsig.persistence import sliding_window
from ripser import ripser

# %% 
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

# %% Choose the function to optimize the slidding window
f = lambda t: np.cos(t) + np.cos(3*t)
# f = lambda t: np.cos(t)
SW = sliding_window(f, bounds=(0, 12*np.pi))

# %% Plot periodic function
from pbsig.persistence import sw_parameters
_, tau_opt = sw_parameters(bounds=(0, 12*np.pi), d=24, L=6)
dom = np.linspace(0, 12*np.pi, 1200)
w = tau_opt * 24

p = figure(width=450, height=225)
p.line(dom, f(dom))
p.rect(x=w/2, y=0, width=w, height=4, fill_alpha=0.20, fill_color='gray', line_alpha=0, line_width=0)
show(p)

# %% Make a slidding window embedding
from pbsig.linalg import pca
from pbsig.color import bin_color
N, M = 80, 25 # num of points, dimension 
X_delay = SW(n=N, d=M, L=6) ## should be the perfect period

pt_color = (bin_color(np.arange(N), "turbo")*255).astype(np.uint8)
p = figure(width=250, height=250, match_aspect=True)
p.scatter(*pca(X_delay, center=True).T, color=pt_color)
show(p)


# %% First: Ensure we have an (N,M) combination that achieves 6 as it's global maxima! 
# Note: This seems not guarenteed unless with have a have enough dimension
N, M = 80, 25 # 120, 25 works 
_, tau_ub = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=4)
_, tau_lb = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=8)
M_opt, tau_opt = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=6)
tau_rng = np.sort(np.append(np.linspace(tau_lb, tau_ub, 120), tau_opt))
max_pers = np.array([np.max(np.diff(ripser(SW(n=N, d=M, tau=tau))['dgms'][1], axis=1)) for tau in tau_rng])

p = figure(width=400, height=250, title="Max persistence H1")
p.line(tau_rng*M, max_pers, color='black')
p.scatter(tau_rng*M, max_pers, size=3, color='black')

show(p)

# %% 
# (56,72), (0,7,36)
E_birth = np.array([sx.flag_filter(pdist(SW(n=N, d=M, tau=tau)))(sx.Simplex([56,72])) for tau in tau_rng])
T_death = np.array([sx.flag_filter(pdist(SW(n=N, d=M, tau=tau)))(sx.Simplex([0,7,36])) for tau in tau_rng])
p.line(tau_rng*M, T_death - E_birth, color='red')
p.scatter(tau_rng*M, T_death - E_birth, size=2, color='red')
show(p)

# working: (120, 25), (120, 32), (120, 40)
# non-working combo's: (120, 60), (140, 60)

# %% Test fixed pairing optimization idea
from pbsig.vis import figure_dgm
dgm1 = ripser(SW(n=N, d=M, tau=tau_opt))['dgms'][1]
H1_pt = dgm1[np.argmax(np.diff(dgm1, axis=1))]
# np.sum(np.isclose(pdist(SW(n=N, d=M, tau=tau_opt)), H1_pt[0], 1e-7))
show(figure_dgm(dgm1))

# %% Somehow try to acquire the simplex pair 
from spirit.apparent_pairs import SpectralRI
X = SW(n=N, d=M, tau=tau_opt)
dX = pdist(X)
RI = SpectralRI(n=N, max_dim=2)
RI.construct(dX, p=0, apparent=True, discard=False, filter="flag")
RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")

RI._D[0] = RI.boundary_matrix(0)
RI._D[1] = RI.boundary_matrix(1)
RI._D[2] = RI.boundary_matrix(2)
# L = RI.lower_left(i=1.5, j=2.0, p=2, deflate=True, apparent=True)

import timeit
timeit.timeit(lambda: RI.query(p=1, a=5.0, b=5.0), number=10) # 0.1448708810057724
timeit.timeit(lambda: RI.query(p=1, a=2.0, b=7.0), number=10) # 9.410425203008344
timeit.timeit(lambda: RI.query(p=1, a=3.0, b=6.0), number=10) # 0.3521803740004543
timeit.timeit(lambda: RI.query(p=1, a=6.0, b=7.0), number=10) # 6.88618484599283
timeit.timeit(lambda: RI.query(p=1, a=1.5, b=2.0), number=10) # 0.01599665600224398
timeit.timeit(lambda: ripser(X), number=10)
## keeping b as low as possible is best

# %% Plot the max persistence 
from pbsig.persistence import ph, pm, validate_decomp, generate_dgm, is_reduced, low_entry
import scipy.sparse as sps

X = SW(n=N, d=M, tau=tau_opt)
# K = sx.rips_filtration(X,p=2)
S = sx.rips_complex(X, p=2)
K = sx.RankFiltration(S, f=sx.flag_filter(pdist(X)))
K.order = 'reverse colex'
D = sx.boundary_matrix(K).tocoo()
D.data = D.data % 2 
D, V = D.astype(np.float32), sps.identity(len(K)).astype(np.float32) # .astype(np.int64)
R, V = pm.phcol(D, V, range(len(K)))  
# np.max(np.abs((D @ V) - R).data) # this should they are the same
# assert validate_decomp(D, R, V, epsilon=1e-2) # 

dgm1 = generate_dgm(K, R, simplex_pairs=True)[1]
# R,V = ph(K, output="RV", engine='cpp', validate=False)
# dgm1 = generate_dgm(K, R, simplex_pairs=True)[1]
dgm1[np.argmax(dgm1['death'] - dgm1['birth'])]['birth']
dgm1[np.argmax(dgm1['death'] - dgm1['birth'])]['death']

from pbsig.vis import figure_dgm
show(figure_dgm(dgm1))

dgm1_true = ripser(SW(n=N, d=M, tau=tau_opt))['dgms'][1]
show(figure_dgm(dgm1_true))

# (56,72), (0,7,36)


# sx.flag_filter(pdist(X_delay))((0, 23))
# sx.flag_filter(pdist(X_delay))((6,14,37))
## Verified the simplex pairs match, just watch the delay variable

# edge = dgm1['creators'][np.argmax(dgm1['death'] - dgm1['birth'])]
# triangle = dgm1['destroyers'][np.argmax(dgm1['death'] - dgm1['birth'])]
# assert np.isclose(np.max(dgm1['death'] - dgm1['birth']), np.max(pdist(X_delay[triangle])) - np.max(pdist(X_delay[edge])))
# print(f"Max pers: {np.max(dgm1['death'] - dgm1['birth']):.4f}")
# opt_max_pers = np.max(dgm1['death'] - dgm1['birth'])

# %% 
from pbsig.persistence import sw_parameters, low_entry
# _, tau_opt = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=6)
_, tau_ub = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=5)
_, tau_lb = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=7)
M_opt, tau_opt = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=6)
tau_rng = np.sort(np.append(np.linspace(tau_lb, tau_ub, 120), tau_opt))




E_birth = np.array([np.max(pdist(SW(n=N, d=M, tau=t)[edge,:])) for t in tau_rng])
T_death = np.array([np.max(pdist(SW(n=N, d=M, tau=t)[triangle,:])) for t in tau_rng])

# np.min((np.abs(opt_max_pers - (T_death - E_birth))))
# assert np.any(np.isclose((np.abs(opt_max_pers - (T_death - E_birth))), 0.0, atol=1e-7))
np.searchsorted(tau_rng, tau_opt)

## Just forget it, the persistence algorithm is too difficult to implement correctly!
# from pbsig.persistence import pHcol
# N = 30
# X_delay = SW(n=N, d=M, tau=tau_rng[0])
# dgm_true = ripser(X_delay)
# K = sx.rips_filtration(X_delay, p=2)
# K = sx.RankFiltration(K)
# D = sx.boundary_matrix(K).tocoo()
# D, V = D.astype(np.float32), sps.identity(len(K)).astype(np.float32) # .astype(np.int64)
# # R, V = pm.phcol(D, V, range(len(K)))  
# # pHcol(D.tolil(), V.tolil())
# assert validate_decomp(D, R, V)
# assert np.all([sx.Simplex(K[r][1]) in sx.Simplex(K[c][1]).boundary() for r,c in zip(D.row, D.col)])
# assert len(R.data) == len((D @ V).data)
# assert np.allclose((R.data - (D @ V).data) % 2, 0)

# assert np.all(r <= c)
dgm_test = generate_dgm(K, R, simplex_pairs=True)[1]

R_test = (D @ V)
R_test.eliminate_zeros()
assert np.sum(low_entry(R) != low_entry(R_test)) == 0

dgm_true[]
dgm_test[0]


np.random.uni



other_tau = 2*np.pi / (6 * (M + 1))
np.max(np.diff(ripser(SW(n=N, d=M, tau=other_tau))['dgms'][1], axis=1))


# %% Get the pair at the optimal step size and determine how relevent its lifetime is globally


