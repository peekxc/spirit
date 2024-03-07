import numpy as np 
import spirit
from scipy.spatial.distance import pdist, squareform, cdist
from combin import rank_to_comb, comb_to_rank
from math import comb
import splex as sx
from spirit import apparent_pairs
from spirit.apparent_pairs import SpectralRI, clique_mod
from math import comb
from combin import comb_to_rank
from spirit.apparent_pairs import clique_mod
from scipy.spatial.distance import pdist 

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

def test_basic_rips():
  X = np.random.uniform(size=(120,2))
  dX = pdist(X)
  RI = SpectralRI(len(X), max_dim = 2)
  RI.construct(dX, filter="flag", apparent=True, discard=False)
  
  from collections import Counter
  from math import comb
  d, (ri,ci) = RI.cm.build_coo(2, RI._simplices[2], RI._simplices[1])
  assert np.max(ri) < comb(len(X), 2) 
  assert np.all(d == np.tile([1, -1, 1], len(RI._simplices[2])))
  from scipy.sparse import coo_array
  D2_test = coo_array((d, (ri,ci)), shape=(len(RI._simplices[1]), len(RI._simplices[2])), dtype=np.float32)
  
  # rank_to_comb(280839, n=len(X), order='colex', k=3)
  # RI.cm.boundary(280839, 2)
  # np.flatnonzero(RI._simplices[1] == 7020) # 119 
  # np.flatnonzero(RI._simplices[1] == 7138) # 1 
  # np.flatnonzero(RI._simplices[1] == 7139) # 0
  ## To test: concat these together and 
  edge_map = {s:i for i,s in enumerate(RI._simplices[1])}
  [edge_map[r] for r in RI.cm.boundary(280839, 2)]

  rank_to_comb(RI._simplices[2][[5]], k=3, n=len(X), order='colex')
  rank_to_comb(RI._simplices[1][ri[ci == 5]], k=2, n=len(X), order='colex')
  RI.cm.boundary()
  # comb(120,2)
  # len(RI._simplices[1])
  # np.sum(RI._status[1] != 0)

  ## Update: Shouldn't discard the positive edges if constructing D2! 
  from combin import rank_to_comb
  from splex.sparse import _fast_boundary
  D2_truth = _fast_boundary(
    rank_to_comb(RI._simplices[2], k=3, n=len(X), order='colex'), 
    rank_to_comb(RI._simplices[1], k=2, n=len(X), order='colex'), 
    dtype=(np.uint32, 3)
  )
  from spirit.apparent_pairs import deflate_sparse


  # sx._fast_boundary(RI._simplices[1], 

  RI.lower_left
  

  ## Summary: 
  ## a. Working memory is a factor -- constructing only non-posiive simplices about 10% faster 
  ## b. Detecting negative simplices is surprising expensive -- about 25% slower!
  ## c. Not detecting pairs at all runs about 10% of the total time than w/ both detection
  import timeit
  timeit.timeit(lambda: RI.cm.build(2, 100.0, False, False, False), number=10)
  timeit.timeit(lambda: RI.cm.build(2, 100.0, False, False, True), number=10)
  timeit.timeit(lambda: RI.cm.build(2, 100.0, True, False, False), number=10)
  timeit.timeit(lambda: RI.cm.build(2, 100.0, True, False, True), number=10)
  timeit.timeit(lambda: RI.cm.build(2, 100.0, True, True, True), number=10)
  timeit.timeit(lambda: RI.cm.build(2, 100.0, False, True, True), number=10)

  # S = sx.rips_complex(X, p = 2)
  
def test_boundary():
  X = np.random.uniform(size=(16,2))
  dX = pdist(X)
  RI = SpectralRI(len(X), max_dim = 2)
  RI.construct(dX, filter="flag", apparent=True, discard=False)
  for tr in RI._simplices[2]:
    t = sx.Simplex(rank_to_comb(tr, k=3, n=len(X), order='colex'))
    E = rank_to_comb(RI.cm.boundary(tr, 2), k=2, n=len(X), order='colex')
    assert np.all([sx.Simplex(e) in t.boundary() for e in E])

def test_coboundary():
  s = (5,3,0)
  n, d = 7, 2
  r = comb_to_rank(s, k=d+1, n=n, order='colex')
  assert r == 13
  cofacets = np.array(clique_mod.enum_coboundary(r, d, n, True))
  assert np.all(cofacets == np.array([28,12,7,6]))
  cofacets_gr = np.array(clique_mod.enum_coboundary(r, d, n, False))
  assert np.all(cofacets_gr == np.array([28]))

def test_coboundary2():
  from itertools import product
  n = 16
  for d, enum_all in product([3,2,1], [True, False]):
    for r in range(comb(n, d+1)):
      s = rank_to_comb(r, k=d+1, n=n, order='colex')
      J = np.setdiff1d(range(n), s) if enum_all else np.array(range(max(s)+1, n))
      s_cofacets = [tuple(sorted(s + (j,))) for j in J]
      s_cofacets_test = np.array(clique_mod.enum_coboundary(r, d, n, enum_all))
      s_cofacets_true = np.flip(np.sort(comb_to_rank(s_cofacets, k=d+1, n=n, order='colex'))) if len(s_cofacets) > 0 else np.array(s_cofacets)
      assert np.allclose(s_cofacets_test, s_cofacets_true)

def test_combined_cofacets():
  n = 10
  R = comb_to_rank([[0,1,2], [1,2,4], [2,4,7]], n=n, order='colex')
  S = rank_to_comb(R, k=3, n=n, order='colex')
  S.sort(axis=1)
  cofacets_0 = np.array(clique_mod.enum_coboundary(R[0], 2, n, True))
  cofacets_1 = np.array(clique_mod.enum_coboundary(R[1], 2, n, True))
  cofacets_2 = np.array(clique_mod.enum_coboundary(R[2], 2, n, True))
  all_cofacets = np.hstack([cofacets_0, cofacets_1, cofacets_2])
  assert len(np.unique(all_cofacets)) < len(all_cofacets)
  clique_mod.enum_coboundary(R[0], 2, n, False)
  clique_mod.enum_coboundary(R[1], 2, n, False)
  clique_mod.enum_coboundary(R[2], 2, n, False)


  C = clique_mod.Cliqueser_star(n, 2)
  C.coboundary(r, 2)


  X = np.random.uniform(size=(16,2))
  dX = pdist(X)
  RI = SpectralRI(len(X), max_dim = 2)
  RI.construct(dX, filter="flag", apparent=True, discard=False)
  for tr in RI._simplices[2]:
    t = sx.Simplex(rank_to_comb(tr, k=3, n=len(X), order='colex'))
    E = rank_to_comb(RI.cm.boundary(tr, 2), k=2, n=len(X), order='colex')
    assert np.all([sx.Simplex(e) in t.boundary() for e in E])


def test_init_basics():
  from math import comb
  from combin import comb_to_rank
  from spirit.apparent_pairs import clique_mod
  from scipy.spatial.distance import pdist 
  X = np.random.uniform(size=(10,2))
  C = clique_mod.Cliqueser_flag(len(X),2)
  S = sx.rips_complex(pdist(X), p=3)
  assert C.n_vertices == 10
  C.init(pdist(X))
  assert np.allclose(np.sort(C.p_simplices(1, np.inf)), np.arange(comb(len(X),2)))

  ## Test all the simplices up the enclosing radius are the same
  pr = np.sort(comb_to_rank(sx.faces(S,p=1)))
  qr = np.sort(comb_to_rank(sx.faces(S,p=2)))
  er = np.sort(C.p_simplices(1, sx.enclosing_radius(X)*2))
  tr = np.sort(C.p_simplices(2, sx.enclosing_radius(X)*2))
  assert np.allclose(er, pr)
  assert np.allclose(tr, qr)


def test_coboundary_gen():
  from itertools import product
  n = 16
  X = np.random.uniform(size=(n,2))
  C = clique_mod.Cliqueser_flag(len(X),2)
  for d in [3,2,1]:
    for r in range(comb(n, d+1)):
      s = rank_to_comb(r, k=d+1, n=n, order='colex')
      J = np.setdiff1d(range(n), s)
      s_cofacets = [tuple(sorted(s + (j,))) for j in J]
      s_cofacets_true = np.flip(np.sort(comb_to_rank(s_cofacets, k=d+1, n=n, order='colex'))) if len(s_cofacets) > 0 else np.array(s_cofacets)
      s_cofacets_test = C.coboundary2(r, d)
      assert np.allclose(s_cofacets_test, s_cofacets_true)

def test_coboundary_union():
  n = 16
  X = np.random.uniform(size=(n,2))
  C = clique_mod.Cliqueser_flag(len(X),2)
  R = np.random.choice(range(n), size=10, replace=False)
  all_cofacets = [C.coboundary2(r, 2) for r in R]
  all_cofacets_true = np.flip(np.unique(np.ravel(all_cofacets)))
  all_cofacets_test = C.cofacets_merged(R,2)
  assert np.allclose(all_cofacets_true, all_cofacets_test)