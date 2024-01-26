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
  X = np.random.uniform(size=(10,2))
  R = apparent_pairs.rip_mod.ripser_lower(pdist(X), 2, 100)
  assert R.threshold == 100.0
  assert R.dim_max == 2
  assert R.n == 10
  assert R.modulus == 2 # this is true

def test_rank_ll():
  X = np.random.uniform(size=(10,2))
  K = sx.rips_filtration(X, p = 2, form = 'rank')
  D2 = sx.boundary_matrix(K, p = 2)

  ## Collect the apparent pairs  
  S = sx.rips_complex(X, p = 2)
  f = sx.flag_filter(pdist(X))
  APs = apparent_pairs.apparent_pairs(S, f, p = 1)
  assert True

def test_paper_example():
  dx = np.array([3,4,5,5,4,3])
  S = sx.rips_complex(dx, p=3)
  DX = squareform(dx)
  R = apparent_pairs.rip_mod.ripser_lower(dx, 2, 2*sx.enclosing_radius(dx))
  index_21 = comb_to_rank([2,1], order='colex', n=sx.card(S,0))
  diam_21 = DX[2,1]
  diam_321 = np.max([DX[3,2], DX[1,2], DX[1,3]])
  index_321 = comb_to_rank([3,2,1], order='colex', n=sx.card(S,0))
  assert np.isclose(diam_21, diam_321)
  
  ## ahhh duality! 
  c = R._apparent_cofacet(index_321, diam_321, 1)
  assert c == index_21
  c = R._apparent_facet(index_21, diam_21, 2)
  assert c == index_321


def test_paper_example2():
  dx = np.array([3,4,5,5,4,3])
  S = sx.rips_complex(dx, p=3)
  for i in range(5):
    RM = apparent_pairs.rip_mod.ripser_lower(dx, 2, 2*sx.enclosing_radius(dx))
    t_ranks = comb_to_rank(sx.faces(S, 2), order='colex', n=sx.card(S,0))
    t_diams = sx.flag_filter(dx)(sx.faces(S,2))
    e_ranks = comb_to_rank(sx.faces(S, 1), order='colex', n=sx.card(S,0))
    e_diams = sx.flag_filter(dx)(sx.faces(S,1))
    
    ## Test apparent facets call
    APs = RM.apparent_facets(t_ranks, t_diams, 2)
    assert np.all(np.flatnonzero(APs != -1) == np.array([2,3]))
    e_verts = rank_to_comb(APs[APs != -1], k=2, order='colex', n = sx.card(S,0))
    assert np.allclose(np.ravel(e_verts), np.array([0,3,1,2]))

    ## Test apparent cofacets call
    APs = RM.apparent_cofacets(e_ranks, e_diams, 1)
    assert np.all(np.flatnonzero(APs != -1) == np.array([2,3]))
    t_verts = rank_to_comb(APs[APs != -1], k=3, order='colex', n = sx.card(S,0))
    assert np.allclose(np.ravel(t_verts), np.array([0,2,3,1,2,3]))


def test_med_rips():
  X = np.random.uniform(size=(20, 2))
  S = sx.rips_complex(X, p=2)
  f = sx.flag_filter(pdist(X))
  # # K = sx.rips_filtration(X, p=2)

  t_diams = f(sx.faces(S,2))
  t_ranks = comb_to_rank(sx.faces(S, 2), order='colex', n = sx.card(S, 0))

  DX = squareform(pdist(X))
  # DX = np.ravel(np.tril(DX))[np.ravel(np.tril(DX)) != 0]
  DX = np.ravel(np.triu(DX))[np.ravel(np.triu(DX)) != 0]

  RM = apparent_pairs.rip_mod.ripser_lower(DX, 2, 4*sx.enclosing_radius(X))
  RM.apparent_facets(t_ranks, t_diams, 2)

  RM._apparent_cofacet(t_ranks[1], t_diams[1], 2)


def test_apparent_facet():
  from combin import rank_to_comb, comb_to_rank
  from spirit.apparent_pairs import clique_mod
  from math import comb
  
  # rank_to_comb(0, k=2, n=10)
  # r = comb_to_rank([5,3,0], order='colex', n=7)
  r = comb(5,3) + comb(3,2) + comb(0,1)
  v = [5,3,0]
  
  idx_below = r
  idx_above = 0
  j = n - 1
  k = 2 + 1
  ii = 0

  while (j >= k and comb(j, k) > idx_below):
    print(f"j = {j}, k = {k}, idx_above={idx_above}, idx_below={idx_below}")
    while (comb(j, k) <= idx_below):
      idx_below -= comb(j, k)
      idx_above += comb(j, k + 1)
      j = j - 1
      k = k - 1
      print(f"idx_below={idx_below}, idx_above={idx_above}, j = {j}, k = {k}")
    cofacet_index = idx_above + comb(j, k + 1) + idx_below

    print(f"cofacet: {cofacet_index}")
    j = j - 1
    idx_below = idx_below - comb(v[ii], k)
    ii += 1
    # idx_below = cofacet_index

  ## Test: (8, 5, 1) w/ n = 13

  comb_to_rank([12,8,5,1], order='colex', n = 13)
  comb_to_rank([11,8,5,1], order='colex', n = 13)
  comb_to_rank([10,8,5,1], order='colex', n = 13)
  comb_to_rank([9,8,5,1], order='colex', n = 13)

  from scipy.special import comb
  # n = 13
  n, d = 7, 2
  sigma = [5,3,0]
  r = comb_to_rank(sigma, order='colex', n = n)
  idx_above, idx_below = 0, r
  # vertices = np.flip([0] + sigma)
  k = d + 1
  v = apparent_pairs.clique_mod.get_max_vertex(r, d+1, n) - 1 # max vertex
  for j in reversed(range(0, n)):
    if j in sigma: 
      continue
    k = (d + 1) - np.searchsorted(-np.array(sigma), -j)
    # while v > j:
    #   k -= 1
    # if k == (len(vertices) - 1):
    #   vertices[k] = j
    # else:
    #   vertices[[k,k+1]] = vertices[[k+1,k]]
    #   vertices[k] = j
    num = 0 if k >= (d+1) else v
    idx_above += comb(num, k+2, exact=True)
    cofacet_rank = idx_above + comb(j, k+1, exact=True) + idx_below
    print(f"{idx_above}, {comb(j, k+1, exact=True)}, {idx_below}")
    print(f"cofacet rank: {cofacet_rank}, cofacet: {rank_to_comb(cofacet_rank, k=d+2, order='colex', n=n)}")
    v = apparent_pairs.clique_mod.get_max_vertex(idx_below, k, n) - 1 # max vertex
    idx_below -= comb(v, k, exact=True) ## post? 


def test_max_vertex():
  n, d = 7, 2
  sigma = [5,3,0]
  r = comb_to_rank(sigma, order='colex', n = n)
  v = apparent_pairs.clique_mod.get_max_vertex(r, d+1, n) - 1 # max vertex
  # Binary searches for the value K satisfying choose(K-1, m) <= r < choose(K, m) 
  assert j == 6
  
## Indeed, it's unclear how ripser code works at all!
def test_ripser_coboundary():
  from scipy.special import comb
  n, d = 7, 2
  sigma = [5,3,0]
  r = comb_to_rank(sigma, order='colex', n = n)
  idx_above, idx_below = 0, r
  j = n - 1
  k = d + 1 
  while j >= k and comb(j, k, exact=True) > idx_below:
    while (comb(j, k, exact=True) <= idx_below):
      idx_below -= comb(j, k, exact=True)
      idx_above += comb(j, k + 1, exact=True)
      --j
      --k
      assert(k != -1)
    cofacet_index = idx_above + comb(j, k + 1) + idx_below
    j -= 1
    print(cofacet_index)
