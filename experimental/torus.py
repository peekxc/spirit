import numpy as np
from scipy.sparse.csgraph import floyd_warshall, minimum_spanning_tree
from scipy.spatial.distance import pdist, cdist, squareform
from set_cover.covers import neighbor_graph_ball, neighbor_graph_knn, neighbor_graph_del
from landmark import landmarks
from ripser import ripser
from bokeh.plotting import show
from bokeh.io import output_notebook
from splex import delaunay_complex
import splex as sx
output_notebook()

# %% Load data set
torus_xyz = np.loadtxt("/Users/mpiekenbrock/spirit/experimental/torus_2k.csv", delimiter=",", skiprows=1)
X = torus_xyz[landmarks(torus_xyz, 30)]
T = delaunay_complex(X)

from combin import rank_to_comb, comb_to_rank
F_vert = np.array(list(sx.faces(T, 1)))
S_vert = np.array(list(sx.faces(T, 2)))
F = np.sort(comb_to_rank(F_vert, k=2, order='colex'))
S = comb_to_rank(S_vert, k=3, order='colex')

# %% Form the oprator 
from comb_laplacian import LaplacianSparse
L = LaplacianSparse(S = S, F = F, n = len(X), k = 3)

# %% 
from sksparse import 
L_sparse = L.tocoo()



# %% TODO: fix to estimate the rank correctly
# from primate.trace import hutch
# hutch(L)






from pbsig.vis import figure_dgm
from bokeh.models import Range1d
# dgms = ripser(X, distance_matrix=False)['dgms']
dgms = ripser(squareform(pdist(X))**3.0, distance_matrix=True)['dgms']
# dgms = ripser(DSP / np.max(DSP), distance_matrix=True)['dgms']
p = figure_dgm(dgms[1])
p.x_range = Range1d(0, 0.35)
p.y_range = Range1d(0, 0.35)
show(p)

# %% 
from scipy.sparse import find
from simplextree import SimplexTree
I, J, weights = find(G)
st = SimplexTree(zip(I,J))
st.expand(2)



st.triangles


# import gudhi
# from gudhi import AlphaComplex
# ac = AlphaComplex(X)
# dgm = np.array([(a,b) for d, (a,b) in dgms if d == 1])
# show(figure_dgm(dgm))









I = np.ravel(D.argpartition(kth = 15, axis=1)[:,:15])
J = np.repeat(np.arange(D.shape[0]),15)



from array import array
I, J = array('I'), array('J')


# D = squareform(pdist(X))
# con_radius = np.max(minimum_spanning_tree(D).data / 2.0)
# enc_radius = np.min(D.max(axis=1))
# r = con_radius + 0.15 * (enc_radius - con_radius)
# G = neighbor_graph_ball(X, radius=r, weighted=True)
# # G = neighbor_graph_knn(X, k=40, weighted=True)
# # G = neighbor_graph_del(X, weighted=True)

# from scipy.sparse.csgraph import floyd_warshall
# DSP = floyd_warshall(G, directed=False)