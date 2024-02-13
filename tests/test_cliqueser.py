import numpy as np 
import spirit
from scipy.spatial.distance import pdist, squareform, cdist
from combin import rank_to_comb, comb_to_rank
import splex as sx
from spirit import apparent_pairs

# dx = np.array([3,4,5,5,4,3])
# S = sx.rips_complex(dx, p=3)
# for i in range(5):
#   RM = apparent_pairs.rip_mod.ripser(dx, 2, 2*sx.enclosing_radius(dx))
#   t_ranks = comb_to_rank(sx.faces(S, 2), order='colex', n=sx.card(S,0))
#   t_diams = sx.flag_filter(dx)(sx.faces(S,2))
#   e_ranks = comb_to_rank(sx.faces(S, 1), order='colex', n=sx.card(S,0))
#   e_diams = sx.flag_filter(dx)(sx.faces(S,1))
  
#   ## Test apparent facets call
#   APs = RM.apparent_facets(t_ranks, t_diams, 2)
#   assert np.all(np.flatnonzero(APs != -1) == np.array([2,3]))
#   e_verts = rank_to_comb(APs[APs != -1], k=2, order='colex', n = sx.card(S,0))
#   assert np.allclose(np.ravel(e_verts), np.array([0,3,1,2]))

def test_init_basics():
  from combin import comb_to_rank
  from spirit.apparent_pairs import clique_mod
  from scipy.spatial.distance import pdist 
  X = np.random.uniform(size=(10,2))
  C = clique_mod.Cliqueser_flag(len(X),2)
  S = sx.rips_complex(pdist(X), p=3)
  assert C.n_vertices == 10
  C.init(pdist(X))
  pr = comb_to_rank(sx.faces(S,p=1))
  qr = comb_to_rank(sx.faces(S,p=2))
  pr


