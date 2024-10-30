# %% 
from typing import *
import scipy as sp
import numpy as np 
import splex as sx
from scipy.spatial.distance import pdist 
from scipy.sparse.linalg import eigsh
from spirit.apparent_pairs import SpectralRI
from pbsig.persistence import sliding_window
from combin import rank_to_comb, comb_to_rank
from ripser import ripser
from math import comb

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
N, M = 50, 20 # num of points, dimension 
X_delay = SW(n=N, d=M, L=6) ## should be the perfect period

pt_color = (bin_color(np.arange(N), "turbo")*255).astype(np.uint8)
p = figure(width=250, height=250, match_aspect=True)
p.scatter(*pca(X_delay, center=True).T, color=pt_color)
show(p)

# %% First: Ensure we have an (N,M) combination that achieves 6 as it's global maxima! 
# Note: This seems not guarenteed unless with have a have enough dimension
from bokeh.models import Span 
_, tau_ub = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=4)
_, tau_lb = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=8)
# M_opt, tau_opt = sw_parameters(bounds = (0, 12 * np.pi), d=M, L=6)
tau_opt = 5.625 / M
tau_rng = np.sort(np.append(np.linspace(tau_lb, tau_ub, 400), tau_opt))
dgms_rng = [ripser(SW(n=N, d=M, tau=tau), coeff=23)['dgms'][1] for tau in tau_rng]
max_pers = np.array([np.max(np.diff(dgm, axis=1)) for dgm in dgms_rng])

p = figure(width=400, height=250, title="Max persistence H1")
p.line(tau_rng*M, max_pers, color='black')
p.scatter(tau_rng*M, max_pers, size=3, color='black')

opt_window = (M*2*np.pi)/(M+1) 
p.add_layout(Span(location=opt_window, dimension='height', line_color='red'))
show(p)

# def is_prime(x):
#   return all(x % i for i in range(2, x))

# def next_prime(x):
#   return min([a for a in range(x+1, 2*x) if is_prime(a)])

# %% Test fixed pairing optimization idea
from pbsig.vis import figure_dgm
dgm1 = ripser(SW(n=N, d=M, tau=tau_rng[0]), coeff=23)['dgms'][1]
# dgm1 = ripser(SW(n=N, d=M, tau=tau_opt))['dgms'][1]
H1_pt = dgm1[np.argmax(np.diff(dgm1, axis=1))]
# np.sum(np.isclose(pdist(SW(n=N, d=M, tau=tau_opt)), H1_pt[0], 1e-7))
show(figure_dgm(dgm1))
a,b,c,d = 1, 3, 6, 7

# %% Somehow try to acquire the simplex pair 
from spirit.apparent_pairs import SpectralRI
from scipy.spatial.distance import squareform

N = 50
X = SW(n=N, d=M, tau=tau_opt)
# X = SW(n=N, d=M, tau=tau_rng[0])
dX = pdist(X)
DX = squareform(dX)
ER = sx.enclosing_radius(DX)
RI = SpectralRI(n=N, max_dim=2)
RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
RI.construct(dX, p=1, apparent=True, discard=False, filter="flag", threshold=2*ER)
RI.construct(dX, p=2, apparent=True, discard=True, filter="flag", threshold=2*ER)

RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
RI.construct(dX, p=1, apparent=True, discard=False, filter="flag", threshold=(a, d))
RI.construct(dX, p=2, apparent=True, discard=True, filter="flag", threshold=(a, d))

RI._D[0] = RI.boundary_matrix(0)
RI._D[1] = RI.boundary_matrix(1)
RI._D[2] = RI.boundary_matrix(2)

# RI.query(1, a,b,c,d, method="cholesky", summands=True)
RI.query(1, a,b,c,d, method="cholesky", summands=False)
assert RI.query(1, a,b,c,d, method="cholesky", summands=False) == 1
# assert RI.query(1, 2.0, 2.2, 6.0, 6.2) == 1

# %% 
dgm1_abcd = RI.query_pairs(1, a, b, c, d, method="cholesky", verbose=True, simplex_pairs=True)
p = figure_dgm(dgm1_abcd)
show(p)
assert np.all(H1_pt[0] == dgm1_abcd['birth'])
assert np.all(H1_pt[1] == dgm1_abcd['death'])

# %% Debugging 
weights_copy = RI._weights.copy()
RI._weights, wrd = RI._index_weights(1)
ai, bi = np.sum(wrd['weight'] < a), np.sum(wrd['weight'] < b)
ci, di = np.sum(wrd['weight'] < c), np.sum(wrd['weight'] < d)
# wrd['rank'] = -wrd['rank']
# wrd_ranking = np.argsort(np.argsort(wrd, order=('weight', 'dim', 'rank')))
# index_weights = { q : wrd_ranking[wrd['dim'] == q] for q in range(3) }
# RI._weights = index_weights

from spirit.query import points_in_box
p = 1
points_in_box(ai,bi,ci,di,lambda i,j,k,l: RI.query(p,i,j,k,l,method='cholesky'), True)
weights_copy[1][RI._weights[1] == 150]
weights_copy[2][RI._weights[2] == 1144]
weights_copy[2][RI._weights[2] == 1110]
print(H1_pt)

## Wrong one yields   (104,106,1039,1889) = 0
## Correct one yields (104,106,1039,1889) = 1

# %% 
validate = True
pairs = [None] * len(tau_rng)
for ii, tau in enumerate(tau_rng):
  X = SW(n=N, d=M, tau=tau)
  dX = pdist(X)
  RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
  RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
  RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")
  RI._D[0] = RI.boundary_matrix(0)
  RI._D[1] = RI.boundary_matrix(1)
  RI._D[2] = RI.boundary_matrix(2)
  pairs[ii] = RI.query_pairs(1, a, b, c, d, method="cholesky", verbose=False, simplex_pairs=True)
  if len(pairs[ii]) > 0 and validate:
    dgm1 = ripser(SW(n=N, d=M, tau=tau))['dgms'][1]
    H1_pt = dgm1[np.argmax(np.diff(dgm1, axis=1))]
    assert np.all(H1_pt[0] == pairs[ii]['birth'])
    assert np.all(H1_pt[1] == pairs[ii]['death'])
  print(ii)

RI.query_pairs(1, a, b, c, d, method="cholesky", verbose=True, simplex_pairs=True)



# RI.cm.simplex_weight(106, 1)
# RI.cm.simplex_weight(1394, 2)

RI._simplices[1][np.flatnonzero(RI._weights[1] == 150)]
RI.cm.simplex_weight(657, 1)

weights_copy[1][RI._simplices[1] == 150]



# %% Use the simplex pairs to create a gradient
from pbsig.persistence import sw_parameters
def sliding_window_np(f: Callable, M: int, N: int, bounds: Tuple = (0, 1)):
  ''' Creates a slidding window point cloud over 'n' windows '''
  def _sw(tau: float, ind = None):
    ind = np.arange(N) if ind is None else np.array(ind)
    d, tau = sw_parameters(bounds, d=M, tau=tau, w=None, L=None)
    T = np.linspace(bounds[0], bounds[1] - d*tau, N)
    return f(T[ind,np.newaxis] + np.arange(d+1)*tau)
  return _sw

def num_deriv_sw(dgm: np.ndarray, p: int, SW: Callable, **kwargs):
  import numdifftools as nd
  if len(dgm) == 0: 
    return 0 
  max_pers_ind = np.argmax(dgm['death'] - dgm['birth'])
  cr = np.ravel(rank_to_comb(dgm[max_pers_ind]['creator'], k=p+1, n=len(X), order='colex'))
  de = np.ravel(rank_to_comb(dgm[max_pers_ind]['destroyer'], k=p+2, n=len(X), order='colex'))
  diam_p = lambda t: np.max(pdist(SW(t, cr)))
  diam_q = lambda t: np.max(pdist(SW(t, de)))
  return nd.Derivative(lambda t: diam_q(t) - diam_p(t), method='central', full_output=True, order=3)


f = lambda t: np.cos(t) + np.cos(3*t)
SW_np = sliding_window_np(f, M, N, bounds=(0, 12*np.pi))
valid_grads = np.flatnonzero(np.array([len(d) for d in pairs]) > 0)
dgm = pairs[30]

a,b,c,d = 1, 2, 6, 7
def get_deriv(tau, ripser: bool = False):
  X = SW_np(tau)
  dX = pdist(X)
  RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
  RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
  RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")
  RI._D[0] = RI.boundary_matrix(0)
  RI._D[1] = RI.boundary_matrix(1)
  RI._D[2] = RI.boundary_matrix(2)
  pairs = RI.query_pairs(1, a, b, c, d, method="cholesky", verbose=False, simplex_pairs=True)
  if len(pairs) == 0:
    return None
  return num_deriv_sw(pairs, 1, SW=SW_np)

# dr = num_deriv_sw(dgm, 1, SW=SW_np)
DR = [get_deriv(tau) for tau in tau_rng]
sw_fun = np.array([dr.fun(tau) if dr is not None else 0 for dr, tau in zip(DR, tau_rng)])

np.flatnonzero(np.logical_and(sw_fun != 0.0, sw_fun != max_pers))
tau = tau_rng[30]

dgm_true = ripser(SW_np(tau=tau))['dgms'][1]
np.max(np.diff(dgm_true, axis=1))
show(figure_dgm(dgm_true))

# np.max(np.diff(ripser(SW(n=N, d=M, tau=tau))['dgms'][1], axis=1))
X = SW_np(tau)
dX = pdist(X)
RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")
RI._D[0] = RI.boundary_matrix(0)
RI._D[1] = RI.boundary_matrix(1)
RI._D[2] = RI.boundary_matrix(2)
pairs = RI.query_pairs(1, a, b, c, d, method="cholesky", verbose=True, simplex_pairs=True)

creator = np.ravel(rank_to_comb(pairs['creator'], order='colex', n=len(X), k=2))
destroyer = np.ravel(rank_to_comb(pairs['destroyer'], order='colex', n=len(X), k=3))

print(f"Creator: {creator} w/ diam = {np.max(pdist(SW_np(tau, creator))):.5f}")
print(f"Destroyer: {destroyer} w/ diam = {np.max(pdist(SW_np(tau, destroyer))):.5f}")

# sw_fun = np.array([dr.fun(tau).item() for tau in tau_rng])
# sw_gradient = np.array([dr(tau)[0].item() for tau in tau_rng])

# %% Show gradient
p = figure(width=350, height=300)
p.scatter(tau_rng*M, sw_fun, color=np.where(sw_gradient > 0, 'red', 'green'))
p.line(tau_rng*M, max_pers, color='black')
p.scatter(tau_rng*M, max_pers, size=3, color='black')
show(p)





# %% 
from sksparse.cholmod import cholesky_AAt
Dp_csc = RI._D[2].tocsc()


## Simplicial seems faster than supernodal, unless order == natural!
# amd and colamd are fastest orderings, and one of them is the default ordering
# the default mode seems to be supernodal! 
# NOTE: about 23% time spent building coo matrices 
# NOTE: about 54% time spent on cholesky
import timeit
timeit.timeit(lambda: cholesky_AAt(Dp_csc, beta=1e-6, mode='simplicial', ordering_method="colamd").D(), number=150)
timeit.timeit(lambda: cholesky_AAt(Dp_csc, beta=1e-6, mode='supernodal', ordering_method="colamd").D(), number=150)
timeit.timeit(lambda: cholesky_AAt(Dp_csc, beta=1e-6).D(), number=150)
F = cholesky_AAt(Dp_csc, beta=1e-6, mode='simplicial')


# %% profiling
from spirit.query import points_in_box, bisection_tree, find_negative
from line_profiler import LineProfiler
profile = LineProfiler()
profile.add_function(RI.query_pairs)
profile.add_function(points_in_box)
profile.add_function(bisection_tree)
profile.add_function(find_negative)
profile.add_function(RI.query)
profile.add_function(RI.rank)
profile.add_function(RI.lower_left)
profile.enable_by_count()
RI.query_pairs(1, a, b, c, d, method="cholesky")
profile.print_stats()



top10 = dgms_index[1][np.argpartition(-(dgms_index[1]['death'] - dgms_index[1]['birth']), 10)][:10]

ai, bi = np.sum(wrd['weight'] < 1.0), np.sum(wrd['weight'] <= 2.0)
ci, di = np.sum(wrd['weight'] < 6.0), np.sum(wrd['weight'] <= 7.0)
RI.query(1,ai,bi,ci,di, method="cholesky", summands=False)
RI.query(1,bi,ci, method="cholesky", summands=False)

from spirit.query import points_in_box, bisection_tree
query = lambda i,j,k,l: RI.query(1,i,j,k,l, method="cholesky", summands=False)
# query = lambda i,j,k,l: RI.query(1,j,k, method="cholesky", summands=False)
points_in_box(ai,bi,ci,di, query=query, verbose=True)


## points in box say the pair is 111, 1570
re = RI._simplices[1][np.flatnonzero(RI._weights[1] == 107)]
rt = RI._simplices[2][np.flatnonzero(RI._weights[2] == 1395)]
diam_f = sx.flag_filter(dX)
pos_edg = np.ravel(rank_to_comb(re, k=2, order='colex', n=len(X)))
neg_tri = np.ravel(rank_to_comb(rt, k=3, order='colex', n=len(X)))

## Compare the pair found to the ground truth
## works if remember to offset weights by 1! 
## Full index persistence works! It's unclear if local index would work too 
diam_f(pos_edg), diam_f(neg_tri)
H1_pt


# %% Indeed tracking the individual pair is not useful
tau_filters = [sx.flag_filter(pdist(SW(n=N, d=M, tau=tau))) for tau in tau_rng]
p = figure(width=400, height=250, title="Max persistence H1")
p.line(tau_rng*M, max_pers, color='black')
p.scatter(tau_rng*M, max_pers, size=3, color='black')

b_edge = np.array([tau_filters[i](pos_edg) for i,tau in enumerate(tau_rng)])
d_tria = np.array([tau_filters[i](neg_tri) for i,tau in enumerate(tau_rng)])
p.scatter(tau_rng*M, d_tria - b_edge, color='red', size=2)
show(p)

## 
tau = 5.625 / M


# %% Get the index persistence 
from spirit.query import _index_persistence
from pbsig.persistence import ph
K = sx.rips_filtration(X, p=2)
K = sx.RankFiltration(K)
K.order = 'reverse colex'
K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
dgms_index = ph(K_index, simplex_pairs=True, validate=False)

  
from pbsig.vis import figure_dgm, show, output_notebook
output_notebook()
p = figure_dgm(dgms_index[1])
p.xaxis[0].ticker = np.arange(len(K))
p.yaxis[0].ticker = np.arange(len(K))
show(p)


# %% Re-weigh using index persistence
# weights = np.hstack([RI._weights[0], RI._weights[1], RI._weights[2]])
# ranks = np.hstack([RI._simplices[0], RI._simplices[1], RI._simplices[2]])
# dims = np.hstack([np.repeat(0, len(RI._simplices[0])), np.repeat(1, len(RI._simplices[1])), np.repeat(2, len(RI._simplices[2]))])
# wrd = np.array(list(zip(weights, ranks, dims)), dtype=[('weight', 'f4'), ('rank', 'i4'), ('dim', 'i4')])
# wrd_ranking = np.argsort(np.argsort(wrd, order=('weight', 'dim', 'rank'))) + 1
# RI._weights[0] = wrd_ranking[dims == 0]
# RI._weights[1] = wrd_ranking[dims == 1]
# RI._weights[2] = wrd_ranking[dims == 2]
# %% Get the actual pair 
from spirit.query import _generate_bfs
N = sum([len(RI._simplices[p]) for p in range(3)])
_generate_bfs(0, N-1)[0]

## Reindex by dimension? Or strictly by weight ?
RI._weights[0] = np.argsort(np.argsort(RI._weights[0])) + 1
RI._weights[1] = np.argsort(np.argsort(RI._weights[1])) + max(RI._weights[0]) + 1
RI._weights[2] = np.argsort(np.argsort(RI._weights[2])) + max(RI._weights[1]) + 1

np.hstack((RI._weights[1], RI._weights[2]))

# RI.query(1, 0, 125, 126, 250)
len(points_in_box(0, 125, 126, 250, query=lambda i,j,k,l: RI.query(1, i,j,k,l, method="cholesky")))




len(RI._simplices[1])
comb(150,3)

## Let's say about 1/16 * C(n,2) is about the threshold needed to consider cofacet merging
import timeit
r_edges = np.random.choice(RI._simplices[1], size=len(RI._simplices[1]) // 16, replace=False)
timeit.timeit(lambda: ripser(X), number=10)
timeit.timeit(lambda: RI.cm.cofacets_merged(RI._simplices[1], 1), number=10)
timeit.timeit(lambda: RI.cm.cofacets_merged(r_edges, 1), number=10)
timeit.timeit(lambda: RI.cm.p_simplices(2, np.inf), number=10)
timeit.timeit(lambda: RI.cm.p_simplices2(2, np.inf), number=10)
## Enumerating union of cofacets about 6x slower than p_simplices 
## Enumerating p_simplices2 is about 2x slower than p_simplices
## Somehow ripser is just barely slower than enumerating all triangles

r_edges = RI._simplices[1][RI._status[1] <= 0]
timeit.timeit(lambda: RI.cm.cofacets_merged(r_edges, 1), number=10)


import timeit
timeit.timeit(lambda: ripser(X), number=10)
timeit.timeit(lambda: RI.construct(dX, p=2, apparent=True, discard=False, filter="flag"), number=10)

## So, if the set of edges E to collect cofacets is in fact the entire set, then the enumeration 
## with all_cofacets = False will indeed enumerate all cofacets once. But that is a big requirement
pos_edges = RI._simplices[1][RI._simplices[1] >= 0]
cofacets_uni = RI.cm.collect_cofacets(pos_edges, 1, np.inf, False, False)
cofacets_all = RI.cm.collect_cofacets(pos_edges, 1, np.inf, False, True)
assert np.allclose(np.flip(np.sort(cofacets_uni)), np.flip(np.unique(cofacets_all)))
len(cofacets_uni)
len(np.unique(cofacets_all))
# RI.construct(dX, p=2, apparent=True, discard=True, threshold=sx.enclosing_radius(dX) * 2, filter="flag")

## There are only 79 non-apparent edges, which means only 79 triangles that are unaccounted for 
## and would contribute to the rank of D2! In contrast, there are C(50, 3) = 19600 triangles in total 
## So less than 1% contribute to the rank! We could prune the 99% in the nullspace... or we could search 
## for the 1% that lie in the Im(D2) (how?)
## We don't need to know exactly those remaining triangles / edge pairs, because that is persistence. 
## We know that of remaining 79 edges E' which are not apparent, |V| of them must be independent if the complex is connected, 
## we can deduce from 0d persistence, and the very small fraction left... we may need to just enumerate all the cofacets 
## below the enclosing radius and then let the rank computation handle the rest (on the)

len(RI._simplices[1]) - np.sum(RI._status[1] > 0)


## Collect the triangles associated with the apparent positive edges
AT = RI._status[1][RI._status[1] > 0]
RT = np.setdiff1d(RI._simplices[2], AT)

RI.boundary_matrix
from spirit.apparent_pairs import boundary_matrix

# np.vstack((triangles[:,[0,1]], triangles[:,[0,2]], triangles[:,[1,2]]))
e_ranks = np.unique(comb_to_rank(np.vstack((triangles[:,[0,1]], triangles[:,[0,2]], triangles[:,[1,2]])), order='colex', n=N))
np.linalg.matrix_rank(boundary_matrix(2, AT, e_ranks).todense())
np.linalg.matrix_rank(boundary_matrix(2, RT, e_ranks).todense())
np.linalg.matrix_rank(boundary_matrix(2, np.hstack((AT, RT)), e_ranks).todense())
# len(AT) == 1146, rank(AT) = 1146
# len(RT) == 428, rank(RT) == 421
# len(AT) + len(RT) = 1574, rank([AT|RT]) = 1146
## Need to detect / filter the triangles chains RT to see if they like in span(AT)

import simplextree as st
TV = np.flipud(rank_to_comb(RT, k=3, order='colex', n=N))
ST = st.SimplexTree(TV)


## Replace edge as node ids -- then also do the same to the triangles! 
## Edit: still would need unrank triangles, which could be slow / non-simd...
## But what about breaking up loops + 
edge_map = { r: i for i,r in enumerate(RI._simplices[1]) }
triangles = rank_to_comb(RI._simplices[2], order='colex', n=N, k=3)
I = np.array([edge_map[e] for e in comb_to_rank(triangles[:,[0,1]], order='colex', n=N, k=2)])
J = np.array([edge_map[e] for e in comb_to_rank(triangles[:,[0,2]], order='colex', n=N, k=2)])
K = np.array([edge_map[e] for e in comb_to_rank(triangles[:,[1,2]], order='colex', n=N, k=2)])


# I = np.array([0,0,1,1,3,3,2,2,2])
# J = np.array([3,2,3,2,2,1,1,1,0])
# x = (np.arange(4) + 1) / 10
# y = np.zeros(4)
# y[I] += x[J]
# x * np.bincount(I)
# x * np.bincount(J)


# def _matvec(x: np.ndarray):
#   # y[I] += x[K] 
#   y = np.zeros(len(x))
#   np.add.at(y, I, x[K])
#   np.add.at(y, K, x[I])
#   np.add.at(y, J, x[I])
#   np.subtract.at(y, I, x[J])
#   np.subtract.at(y, K, x[J])
#   np.subtract.at(y, J, x[K])
#   return y
# x = np.random.uniform(size=len(edge_map))

e_ids, degree = np.unique(np.hstack([I,J,K]), return_counts=True)
degree_vec = np.zeros(len(edge_map))
degree_vec[e_ids] = degree

# for i,j,k in rank_to_comb(RI._simplices[2], order='colex', n=N, k=3):x/


# ST.expand(2)
# ST

# ripser(X, maxdim=1)['num_edges']
## len(RI._simplices[1]) == 124750
## ripser # edges:          124541
## Suppose an edge 'e' is detected as positive
## => R_1(e) == 0, that is e forms a zero column in the reduced matrix 
## => e does not contirbute to rank(R_1)
## => there exists a destroyer/negative 2-simplex which *does* contribute to rank(R_2) (w/ exactly +1)
## => ?? that triangle auto increases rank R2, but otherwise doesn't need to be counted / included in the query model ??
## => That triangle forms a pivot in R2
## 
## Can I get a subcomplex of smaller size plz? no way ripser is doing this
## Suppose I just take the complex formed by the unknown edges? 
## 124541

17634944 / comb(500,3)

x = np.arange(17634944)
np.sum(x)

RI._D[0] = RI.boundary_matrix(0)
RI._D[1] = RI.boundary_matrix(1)
RI._D[2] = RI.boundary_matrix(2)

# L = RI.lower_left(i=1.5, j=2.0, p=2, deflate=True, apparent=True)

weights = np.hstack([RI._weights[1], RI._weights[2]])
ranks = np.hstack([RI._simplices[1], RI._simplices[2]])
dims = np.hstack([np.repeat(1, len(RI._simplices[1])), np.repeat(2, len(RI._simplices[2]))])
wrd = np.array(list(zip(weights, ranks, dims)), dtype=[('weight', 'f4'), ('rank', 'i4'), ('dim', 'i4')])
wrd_ranking = np.argsort(np.argsort(wrd, order=('weight', 'dim', 'rank')))
weights_sorted = weights[np.argsort(wrd_ranking)]

index_map = dict(zip(wrd_ranking, wrd['weight']))
inv_index_map = lambda x: np.searchsorted(weights_sorted, x)

def query_oracle(i: int, j: int, k: int, l: int) -> int:
  return RI.query(1, index_map[i], index_map[j], index_map[k], index_map[l], method="cholesky")

from spirit.query import points_in_box, bisection_tree, midpoint
i,j,k,l = inv_index_map([1.0, 2.0, 7.0, 8.0])
mu_init = query_oracle(i,j,k,l)

positive = {}
bisection_tree(i, j, k, l, mu_init, query_oracle, positive, verbose=True)

## Fix this
query_oracle(49,50,3076,4998) # 1
query_oracle(49,49,3076,4998) # 0
query_oracle(50,50,3076,4998) # 0

# points_in_box(33, 250, 3076, 4998, query=query_oracle)


# %% Hacking ripser to get the simplex pairs?
from scipy.spatial.distance import squareform
# assert comb(len(X), 2) == len(np.unique(pdist(X))) # general position might be required here
index_dist = np.argsort(np.argsort(pdist(X)))
dgms_mod2 = ripser(squareform(index_dist), distance_matrix=True)['dgms']

H1_index_birth = dgms_mod2[1][:,0]
H1_index_death = dgms_mod2[1][:,1]

def query_oracle(i: int, j: int, k: int, l: int) -> int:
  in_birth = np.logical_and(i <= H1_index_birth, H1_index_birth <= j)
  in_death = np.logical_and(k <= H1_index_death, H1_index_death <= l)
  return np.sum(np.logical_and(in_birth, in_death))

from spirit.query import _generate_bfs
all_boxes = _generate_bfs(0, comb(len(X), ))
all_pairs = { k : points_in_box(*box, query_oracle) for k, box in all_boxes.items() }
all_pairs = np.array([np.ravel(tuple(p.items())) for p in all_pairs.values() if len(p) > 0])
all_pairs = all_pairs[np.lexsort(np.rot90(all_pairs)),:]

points_in_box(33, 250, 3076, 4998, query=query_oracle)

# %% 
import timeit
# timeit.timeit(lambda: RI.query(p=1, a=5.0, b=5.0, method="direct"), number=10) # 0.1448708810057724
# timeit.timeit(lambda: RI.query(p=1, a=2.0, b=7.0, method="direct"), number=10) # 9.410425203008344
# timeit.timeit(lambda: RI.query(p=1, a=3.0, b=6.0, method="direct"), number=10) # 0.3521803740004543
# timeit.timeit(lambda: RI.query(p=1, a=6.0, b=7.0, method="direct"), number=10) # 6.88618484599283
# timeit.timeit(lambda: RI.query(p=1, a=1.5, b=2.0, method="direct"), number=10) # 0.01599665600224398
timeit.timeit(lambda: RI.query(p=1, a=5.0, b=5.0, method="cholesky"), number=10) # 0.0216086720029125
timeit.timeit(lambda: RI.query(p=1, a=2.0, b=7.0, method="cholesky"), number=10) # 0.08865450200391933
timeit.timeit(lambda: RI.query(p=1, a=3.0, b=6.0, method="cholesky"), number=10) # 0.03193637100048363
timeit.timeit(lambda: RI.query(p=1, a=6.0, b=7.0, method="cholesky"), number=10) # 0.08437441800197121
timeit.timeit(lambda: RI.query(p=1, a=1.5, b=2.0, method="cholesky"), number=10) # 0.010919381995336153
timeit.timeit(lambda: RI.query(p=1, a=0.0, b=1.0, c=6.0, d=7.0, method="cholesky"), number=10) # 0.53
timeit.timeit(lambda: ripser(X), number=10)                                      # 0.29878249199828133
## keeping b as low as possible seems quite good 

def query_trace():
  return RI.query(p=1, a=0.0, b=1.0, c=6.0, d=7.0, method="trace", maxiter=10, deg=5, gap=1e-5, atol=0.5, summands=False)

timeit.timeit(lambda: query_trace(), number=10) 

# %% Determine the bottlenecks
from spirit.apparent_pairs import compress_index, deflate_sparse
from line_profiler import LineProfiler
profile = LineProfiler()
profile.add_function(RI.query)
profile.add_function(RI.rank)
profile.add_function(RI.lower_left)
profile.add_function(compress_index)
profile.enable_by_count()
RI.query(p=1, a=1.0, b=2.0, c=7.0, d=8.0, method="cholesky")
profile.print_stats()

## %% Attempt to find the actual pair 


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

import yep
yep.start('enum_triangles.prof')
for i in range(120):
  RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")
yep.stop()


other_tau = 2*np.pi / (6 * (M + 1))
np.max(np.diff(ripser(SW(n=N, d=M, tau=other_tau))['dgms'][1], axis=1))


# %% Get the pair at the optimal step size and determine how relevent its lifetime is globally
G1 = np.random.normal(size=(50,2), loc=(5,5), scale=1.5)
G2 = np.random.normal(size=(50,2), loc=(-5,-5), scale=1.5)
N = np.random.uniform(size=(25,2), low=-5, high=5)
G = np.vstack([G1, G2, N])

p = figure(width=300, height=300)
p.scatter(*G.T)
show(p)

X = np.c_[G, np.ones(len(G))]
w = np.array([-0.5,-0.25,0])

p = figure(width=300, height=300)
p.scatter(*G.T, color=np.where(np.sign(w @ X.T) <= 0, "red", "blue"))
show(p)

y = np.sign(w @ X.T)
Xy = np.c_[G, y]

np.savetxt("perceptron.txt", Xy[np.random.choice(range(len(X)), replace=False, size=len(Xy)),:], fmt="%g,%g,%d")
