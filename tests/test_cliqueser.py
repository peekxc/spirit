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
from pbsig.persistence import ph, low_entry
from combin import comb_to_rank

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


def test_neg_triangle_superset_logic():
  from spirit.apparent_pairs import SpectralRI
  # seed 43
  for i in range(100):
    np.random.seed(i)
    X = np.random.uniform(size=(9,2))
    dX = pdist(X)
    RI = SpectralRI(n=len(X), max_dim=2)
    RI.construct(dX, p=0, apparent=True, discard=False, filter="flag")
    RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
    RI.construct(dX, p=2, apparent=True, discard=False, filter="flag")

    RI._D[0] = RI.boundary_matrix(0)
    RI._D[1] = RI.boundary_matrix(1)
    RI._D[2] = RI.boundary_matrix(2)

    K = sx.rips_filtration(dX, radius=np.inf, p=2)
    K = sx.RankFiltration(K)
    K.order = 'reverse colex'
    R,V = ph(K, output="RV")
    dims = np.array([sx.dim(s) for i,s in K])
    E_ranks = comb_to_rank(list(map(sx.Simplex, sx.faces(K,1))), n=sx.card(K,0), order='colex')
    T_ranks = comb_to_rank(list(map(sx.Simplex, sx.faces(K,2))), n=sx.card(K,0), order='colex')

    # (sx.boundary_matrix(K) @ V) - R
    from pbsig.persistence import validate_decomp
    assert validate_decomp(sx.boundary_matrix(K), R, V)

    ## Test that the apparent pairs are indeed pivot pairs in R1
    R1 = R[:,dims == 1][dims==0]
    piv_true = low_entry(R1)
    E_pivot_status = {s : s_status  for s, s_status in zip(RI._simplices[1], RI._status[1])}
    piv_test = np.array([E_pivot_status[e] for e in E_ranks])
    assert np.all(piv_true[piv_test > 0] == -1)

    ## Test that the apparent pairs are indeed pivot pairs in R2
    R2 = R[:,dims == 2][dims==1]
    piv_true = low_entry(R2)
    T_pivot_status = {s : s_status  for s, s_status in zip(RI._simplices[2], RI._status[2])}
    piv_test = np.array([T_pivot_status[t] for t in T_ranks])
    assert np.all(piv_true[piv_test > 0] == -1)

    ## Test that all triangles that kill edges lie in the cofacets of the positive edges
    ## Even this might not be guarenteed to be true? 
    neg_triangles = T_ranks[low_entry(R2) != -1]
    pos_edges = E_ranks[low_entry(R1) == -1]
    PE_cofacets = RI.cm.cofacets_merged(pos_edges, 1)
    assert len(np.intersect1d(PE_cofacets, neg_triangles)) == len(neg_triangles)

    ## Test that all triangles that kill edges lie in one of two sets: 
    ## 1. The negative triangles of the apparent (edge, triangle) pairs
    ## 2. The cofacets of the positive non-apparent edges 
    ## Ah: 2 is not necessarily true, because non-apparent edge pairs need not be killed by their cofacets
    ## What if I knew exactly the 
    ## edges past a, triangles <= d 
    neg_ap_triangles = RI._status[1][RI._status[1] > 0]
    unknown_edges = RI._simplices[1][RI._status[1] == 0]
    nap_triangles = RI.cm.cofacets_merged(unknown_edges, 1)

    neg_triangles_true = T_ranks[low_entry(R2) != -1]
    neg_triangles_test = np.union1d(neg_ap_triangles, nap_triangles)
    unknown_triangles = np.setdiff1d(neg_triangles_true, neg_triangles_test)
    assert len(unknown_triangles) == 0, "Negative triangle superset logic failed"

    ## Investigate the unknown triangles 
    assert 66 not in RI._simplices[1] # thus its not an apparent cofacet 
    assert RI.cm.apparent_zero_facet(66, 2) == -1 # double-check its not an apparent cofacet

    ## check out which edge it killed 
    e_unknown = E_ranks[low_entry(R2[:,T_ranks == 66])]
    assert e_unknown not in unknown_edges

    assert e_unknown not in RI._simplices[1][RI._status[1] > 0], "edge is literally a positive simplex"
    t_ap_cofacet = RI._status[1][RI._simplices[1] == e_unknown]

    ## The cofacet search suggests (11, 67) is an apparent pair, but the reduction suggests its (11, 66)
    ## But the E_pivot_check was satified
    RI.cm.apparent_zero_cofacet(11, 1)
    RI.cm.apparent_zero_facet(66,2)
    RI.cm.apparent_zero_facet(67,2)
    assert E_pivot_status[11] == 67
    low_entry(R2)[T_ranks == 66]

    ## Sanity check: check cofacets of 11, find the apparent one
    ## Whoa, there is no 66 in the cofacets of 11...
    ## 11 -> (1,5)
    ## 66 -> (0,5,8)
    ## 67 -> (1,5,8)
    ## So 67 is indeed the first cofacet with diam >= f(11)
    ## reduction says 66 is a pivot though 
    diam_f = sx.flag_filter(dX)
    edge = rank_to_comb(11, k=2, n=len(X), order='colex')
    c_indices = np.setdiff1d(range(len(X)), edge)
    cofacets_c = np.array([np.sort(edge + (c,)) for c in c_indices])
    cofacets_r = comb_to_rank(cofacets_c, k=3, n=len(X), order='colex')
    cofacets_c = cofacets_c[np.argsort(-cofacets_r)]
    cofacets_r = cofacets_r[np.argsort(-cofacets_r)]
    assert np.all(comb_to_rank(cofacets_c, k=3, n=len(X), order='colex') == cofacets_r)
    birth_time = diam_f(edge)
    death_times = diam_f(cofacets_c)

    ## Ensure the filtration is reverse-colexicographically filtered
    t_diams = diam_f(sx.faces(K,2))
    t_ranks = comb_to_rank(sx.faces(K,2), order='colex', n=len(X))
    for diam in np.unique(t_diams):
      class_ind = np.flatnonzero(t_diams == diam)
      if len(class_ind) == 1: 
        continue 
      assert np.all(np.argsort(np.flip(t_ranks[class_ind])) == np.arange(len(class_ind)))

    ## Let's ensure they're symmetric
    assert RI.cm.apparent_zero_cofacet(11, 1) == 67
    assert RI.cm.apparent_zero_facet(67, 2) == 11
    rank_to_comb(T_ranks, k=3, n=len(X), order='colex')

    ## Let's check out the boundary matrix
    ## tri 66 is at index 9, is not apparent, as expected 
    ## edge 11 is at index 23, which is far enough that there's no way 66 touches it, because low(66) == 16
    ## R2.todense()[:24,:10] % 2
    ## so low(< triangle 66 >) == < index 11 > but not the edge rank 11...

def test_neg_triangle_superset_logic():
  theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
  X = np.c_[np.cos(theta), np.sin(theta)]
  dX = pdist(X)
  RI = SpectralRI(n=len(X), max_dim=2)
  RI.construct(dX, p=0, apparent=True, discard=False, filter="flag")
  RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
  RI.construct(dX, p=2, apparent=True, discard=False, filter="flag")
  RI.cm.dgm0(np.inf)

  RI._D[0] = RI.boundary_matrix(0)
  RI._D[1] = RI.boundary_matrix(1)
  RI._D[2] = RI.boundary_matrix(2)
  
  ## Test that all triangles that kill edges lie in the cofacets of the positive edges
  ## Even this might not be guarenteed to be true? 
  neg_triangles = RI._simplices[2][RI._status[2] <= 0]
  pos_edges = RI._simplices[1][RI._status[1] >= 0]
  PE_cofacets = RI.cm.cofacets_merged(pos_edges, 1)
  assert len(np.intersect1d(PE_cofacets, neg_triangles)) == len(neg_triangles)

  ## Try to restrict pos_edges even more
  apparent_triangles = RI.cm.apparent_zero_cofacets(pos_edges, 1)
  pos_nap_edges = pos_edges[apparent_triangles == -1]
  negz_edges = RI._simplices[1][RI._status[1] <= 0]
  other_edges = np.flip(np.union1d(negz_edges, pos_nap_edges))
  assert len(np.intersect1d(RI.cm.cofacets_merged(other_edges, 1), neg_triangles)) == len(neg_triangles)

def test_neg_triangle_superset_logic2():
  theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
  X = np.c_[np.cos(theta), np.sin(theta)]
  dX = pdist(X)
  RI = SpectralRI(n=len(X), max_dim=2)
  RI.construct(dX, p=0, apparent=True, discard=False, filter="flag")
  RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
  RI.construct(dX, p=2, apparent=True, discard=False, filter="flag")

  RI._D[0] = RI.boundary_matrix(0)
  RI._D[1] = RI.boundary_matrix(1)
  RI._D[2] = RI.boundary_matrix(2)

  ## Test that all triangles that kill edges lie in the cofacets of the positive edges
  ## Even this might not be guarenteed to be true? 
  neg_triangles = RI._simplices[2][RI._status[2] <= 0]
  pos_edges = RI._simplices[1][RI._status[1] >= 0]
  PE_cofacets = RI.cm.cofacets_merged(pos_edges, 1)
  assert len(np.intersect1d(PE_cofacets, neg_triangles)) == len(neg_triangles)

  ## Try to restrict pos_edges even more
  apparent_triangles = RI.cm.apparent_zero_cofacets(pos_edges, 1)
  pos_nap_edges = pos_edges[apparent_triangles == -1]
  # negz_edges = RI._simplices[1][RI._status[1] <= 0]
  # other_edges = np.flip(np.union1d(negz_edges, pos_nap_edges))
  # assert len(np.intersect1d(RI.cm.cofacets_merged(other_edges, 1), neg_triangles)) == len(neg_triangles)

  neg_nap_triangles = RI.cm.cofacets_merged(pos_nap_edges, 1)
  nap_edge_status = np.array([RI.cm.apparent_zero_facet(t, 2) for t in neg_nap_triangles])
  neg_nap_triangles2 = neg_nap_triangles[nap_edge_status == -1]
  
  # neg_nap_triangles_2 = RI.cm.cofacets_merged(nap_edge_status[nap_edge_status >= 0], 1)
  # pos_nap_faces = np.unique(np.ravel([RI.cm.boundary(r,2) for r in neg_nap_triangles]))
  
  # neg_nap_triangles_2 = RI.cm.cofacets_merged(pos_nap_faces, 1)
  neg_nap_triangles_3 = np.unique(np.union1d(
    np.unique(apparent_triangles[apparent_triangles != -1]),
    neg_nap_triangles_2 
  ))
  print(f"# neg triangles: {len(neg_triangles)}, UB: {len(neg_nap_triangles_3)}, max: {len(PE_cofacets)}")
  assert len(np.intersect1d(neg_nap_triangles_3, neg_triangles)) == len(neg_triangles)


  # neg_vertices = RI.cm.dgm0(np.inf)
  # np.array([v[0] for v in neg_vertices])



  ## Exact 
  destroyer_tri = rank_to_comb(5, order='colex', k=3, n=len(X))
  birth_edges = rank_to_comb([0,11], order='colex', k=2, n=len(X))
  
  pos_ap_edges = np.array([1,3,4,6,7,8,12,13], dtype=np.int64)
  pos_nap_edges = np.array([0,11], dtype=np.int64)
  neg_edges = np.setdiff1d(np.arange(15), np.union1d(pos_ap_edges, pos_nap_edges))
  assert len(pos_ap_edges) + len(pos_nap_edges) + 5 == comb(6,2)

  ## The cofacets of all positive edges contains the negative triangles somehow
  assert len(np.intersect1d(neg_triangles, RI.cm.cofacets_merged(np.union1d(pos_ap_edges, pos_nap_edges), 1))) == 12
  
  neg_ap_triangles = RI.cm.apparent_zero_cofacets(pos_ap_edges, 1)
  nap_cofacets = RI.cm.cofacets_merged(pos_nap_edges, 1)
  np.union1d(neg_ap_triangles, nap_cofacets)
  np.sort(neg_triangles)
  # np.intersect1d(neg_triangles, 


  import bokeh
  from bokeh.io import output_notebook
  output_notebook()
  from bokeh.plotting import figure, show
  
  p = figure(width=300, height=300)
  p.scatter(*X.T)
  p.text(*X.T, text=np.arange(len(X)), y_offset=-0.01, x_offset=0)
  show(p)


