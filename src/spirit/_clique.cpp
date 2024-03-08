#include <vector> 
#include <cinttypes> 
#include <iostream>
#include <iterator>
#include <concepts> 
#include <queue>
#include <type_traits>
#include <numeric>

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <iterator_facade.h>
#include <combin/combinatorial.h>
#include <disjoint_set.h> 


using index_t = std::int64_t;
using combinatorial::BinomialCoefficientTable;
using combinatorial::unrank_colex;
using Eigen::Dynamic; 
using Eigen::Ref; 
using Eigen::MatrixXf;
using Eigen::MatrixXd;

template< typename F >
using Vector = Eigen::Matrix< F, Dynamic, 1 >;

template< typename F >
using Array = Eigen::Array< F, Dynamic, 1 >;

template< typename F >
using DenseMatrix = Eigen::Matrix< F, Dynamic, Dynamic >;

// template < typename T >
// concept NumericType= std::integral<T> || std::floating_point<T>;

// template <typename T>
// concept DefaultConstructible = std::is_default_constructible_v<T>;

// An SimplexIndexer is something that takes as input a simplex (of some input) and 
// supports computing the index of the simplex in some larger filtration (optionally stored)
template< typename T, typename F = typename T::value_type >
concept SimplexIndexer = requires(T filtration, index_t* a, index_t* b)
{ 
  { filtration.simplex_index(a,b) } ->  std::convertible_to< F >;
  // { T::simplex_index() } -> I;
} && std::constructible_from< T, const size_t >;


template< typename T >
struct StarFilter {
  using value_type = T;

  const size_t n; 
  std::vector< T > weights; 

  StarFilter(const size_t n_) : n(n_) {};
  StarFilter(const size_t n_, std::vector< T > w_) : n(n_), weights(w_) {};

  // template< std::integral I >
  template< typename Iter >
  auto simplex_index(Iter b, Iter e) const -> T {
    T s_weight = -std::numeric_limits< T >::infinity(); 
    for (; b != e; ++b){
      s_weight = std::max(s_weight, weights[*b]);
    }
    return s_weight; 
  }

  template< typename Lambda > 
  void p_simplices(const size_t p, const T threshold, Lambda&& f) const {
    auto vertices = std::vector< size_t >(n, 0);
    std::iota(vertices.rbegin(), vertices.rend(), 0);
    combinatorial::for_each_combination(vertices.begin(), vertices.begin()+(p+1), vertices.end(), [&](auto b, auto e){
      auto s_weight = simplex_index(b, e);
      if (s_weight <= threshold){
        f(b, e, s_weight);
      }
      return false; 
    });
  }
};

template< typename T >
struct FlagFilter {
  using value_type = T;
  
  const size_t n; 
  std::vector< T > weights; 

  FlagFilter(const size_t n_) : n(n_) {}
  FlagFilter(const size_t n_, std::vector< T > w_) : n(n_), weights(w_) {
    assert(w_.size() == (n*(n - 1) / 2));
  };

  // template< std::integral I >
  template< typename Iter >
  auto simplex_index(Iter b, Iter e) const -> T {
    T s_weight = std::numeric_limits< T >::min();
    if (std::distance(b, e) <= 1){ return s_weight; }
    combinatorial::for_each_combination(b, b+2, e, [this, &s_weight](auto b, [[maybe_unused]] auto e){
      s_weight = std::max(
        s_weight, 
        weights.at(combinatorial::rank_lex_2(*b, *(b+1), n))
      ); 
      return false; 
    });
    return s_weight;
  }

  template< typename Lambda > 
  void p_simplices(const size_t p, const T threshold, Lambda&& f) const{
    auto vertices = std::vector< size_t >(n, 0);
    std::iota(vertices.rbegin(), vertices.rend(), 0);
    if (p == 1){
      index_t i = 0; 
      combinatorial::for_each_combination(vertices.begin(), vertices.begin()+2, vertices.end(), [&](auto b, auto e){
        auto s_weight = weights[i++]; // simplex_index(b, e);
        if (s_weight <= threshold){
          f(b, e, s_weight);
        }
        return false; 
      });
    } else {
      combinatorial::for_each_combination(vertices.begin(), vertices.begin()+(p+1), vertices.end(), [&](auto b, auto e){
        auto s_weight = simplex_index(b, e);
        if (s_weight <= threshold){
          f(b, e, s_weight);
        }
        return false; 
      });
    }
  }
};


template< typename T >
struct MetricFilter {
  using value_type = T;
  
  const size_t n; 
  size_t d; 
  std::vector< T > points; 

  MetricFilter(const size_t n_) : n(n_) {}
  MetricFilter(const size_t n_, const size_t d_, std::vector< T > coords) : n(n_), d(d_), points(coords) {};

  // template< std::integral I >
  template< typename Iter >
  auto simplex_index(Iter b, Iter e) const -> T {
    T s_weight = std::numeric_limits< T >::min();
    combinatorial::for_each_combination(b, b+2, e, [this, &s_weight](auto b, [[maybe_unused]] auto e){
      const index_t i = *b;
      const index_t j = *(b+1);
      const T dist_ij = std::inner_product(
		    points.begin()+(i * d), points.begin()+(i+1 * d), points.begin()+(j * d), 0.0, std::plus< value_type >(),
		    [](auto u, auto v) { return (u - v) * (u - v); }
      );
      s_weight = std::max(s_weight, dist_ij); 
      return false; 
    });
    return s_weight;
  }

  template< typename Lambda > 
  void p_simplices(const size_t p, const T threshold, Lambda&& f) const {
    auto vertices = std::vector< size_t >(n, 0);
    std::iota(vertices.rbegin(), vertices.rend(), 0);
    combinatorial::for_each_combination(vertices.begin(), vertices.begin()+(p+1), vertices.end(), [&](auto b, auto e){
      auto s_weight = simplex_index(b, e);
      if (s_weight <= threshold){
        f(b, e, s_weight);
      }
      return false; 
    });
  }

  // Given index of simplex S, its weight, and the vertex id 'v' of face F = S - {v}
  // auto facet_weight(){

  // }

};

using combinatorial::BC; // use pre-allocated BC table

// Enumerates the facets on the boundary of 'simplex'
template< typename Lambda > 
void enum_boundary(const index_t n, const index_t simplex, const index_t dim, Lambda&& f) {
  index_t idx_below = simplex;
  index_t idx_above = 0; 
  index_t j = n - 1;
  bool cont_enum = true; 
  for (index_t k = dim; k >= 0 && cont_enum; --k){
    j = combinatorial::get_max_vertex< false >(idx_below, k + 1, j) - 1; // NOTE: Danger!
    index_t c = BC.at(j, k + 1);
    index_t face_index = idx_above - c + idx_below;
    idx_below -= c;
    idx_above += BC.at(j, k);
    cont_enum = f(face_index);
  }
}

// Enumerates the cofacets on the coboundary of 'simplex'
template< bool all_cofacets = true, typename Lambda > 
void enum_coboundary(const index_t simplex, const index_t dim, const index_t n, Lambda&& f) {
  index_t idx_below = simplex;
  index_t idx_above = 0;
  index_t j = n - 1;
  index_t k = dim + 1;
  bool cont_enum = true;
  if constexpr (all_cofacets){
    while (j >= k && cont_enum){
      // std::cout << "j = " << j << ", k = " << k << std::endl;
      while ((static_cast< index_t >(BC.at(j,k)) <= idx_below)) {
        idx_below -= BC.at(j, k);
        idx_above += BC.at(j, k + 1);
        --j;
        --k;
        //assert(k != -1);
      }
      index_t cofacet_index = idx_above + BC.at(j--, k + 1) + idx_below;
      cont_enum = f(cofacet_index);
    }
  } else {
    while (j >= k && BC(j, k) > size_t(idx_below) && cont_enum){
      // std::cout << "j = " << j << ", k = " << k << std::endl;
      while ((static_cast< index_t >(BC.at(j,k)) <= idx_below)) {
        idx_below -= BC.at(j, k);
        idx_above += BC.at(j, k + 1);
        --j;
        --k;
        //assert(k != -1);
      }
      index_t cofacet_index = idx_above + BC.at(j--, k + 1) + idx_below;
      cont_enum = f(cofacet_index);
    }
  }
}

// Comparator for (i,j) < (k,l) if j < l, and i < k otherwise
template< typename T >
using IndexPair = std::pair< index_t, T >;

template< typename T >
struct LessPair {
  bool operator()(const IndexPair< T >& p1, const IndexPair< T >& p2) const { 
    return p1.second == p2.second ? p1.first < p2.first : p1.second < p2.second;
  }
};

template< typename T >
struct GreaterPair {
  bool operator()(const IndexPair< T >& p1, const IndexPair< T >& p2) const { 
    return p1.second == p2.second ? p1.first < p2.first : p1.second > p2.second;
  }
};

// Set union on K sorted ranges (iterated by generators)
template< typename Gen, typename Lambda > 
void merge_k_sorted(vector< Gen >& K_gens, Lambda&& f){
  using T = typename Gen::value_type; 
  
  // Contains pairs (< index >, < iter >)
  // Add the top element of each generator to the max heap 
  const size_t K = K_gens.size();
  auto heap_data = std::vector< IndexPair< T > >();
  heap_data.reserve(K);
  for (auto i = 0; i < K; ++i){
    if (K_gens[i].has_next()){
      heap_data.push_back(std::make_pair(i, K_gens[i].next()));
    }
  } 
  // < IndexPair< T >,  std::vector< IndexPair< T > >, LessPair< T > >
  // Construct the initial heap (this does in O(K) time)
  auto max_heap = std::priority_queue(heap_data.begin(), heap_data.end(), LessPair< T >());


  // Enumerate through the sorted lists via the heap, 
  T last = -1; 
  while (!max_heap.empty()){
    auto p = max_heap.top();
    max_heap.pop();

    const index_t i = p.first; 
    T val = p.second;
    if (val != last){
      f(val);
      last = val;
    } 

    // If p has successor, advance
    if (K_gens[i].has_next()){
      p.second = K_gens[i].next();
      max_heap.push(p);
    }
  }
}


struct CoboundaryGenerator {
  using value_type = index_t; 

  index_t idx_below = 0;
  index_t idx_above = 0;
  index_t j = 0;
  index_t k = 0;

  CoboundaryGenerator(const index_t simplex, const index_t dim, const index_t n)
  : idx_below(simplex), idx_above(0), j(n - 1), k(dim + 1) {
    if (BC.BT.size() <= size_t(dim + 2) || BC.BT.at(0).size() < n){ 
      BC.precompute(n, dim + 2); 
    } 
  }
  
  // Resets the generator to the given simplex
  void init(const index_t simplex, const index_t dim, const index_t n){
    idx_below = simplex;
    idx_above = 0; 
    j = n - 1; 
    k = dim + 1;
  }

  bool has_next(){ return j >= k; }

  index_t next(){
    while ((static_cast< index_t >(BC.at(j,k)) <= idx_below)) {
      idx_below -= BC.at(j, k);
      idx_above += BC.at(j, k + 1);
      --j;
      --k;
      //assert(k != -1);
    }
    return idx_above + BC.at(j--, k + 1) + idx_below;
  }
};



// 
// void build(const index_t dim_max = 1){
//   std::vector< index_t > simplices, columns_to_reduce;
	
//   // post: simplices contains all sorted edges <= threshold 
//   // post: columns to reduce contains all non-connecting and non-apparent edges 
//   compute_dim_0_pairs(simplices, columns_to_reduce);

//   for (index_t dim = 1; dim <= dim_max; ++dim) {
//     // entry_hash_map pivot_column_index;
//     // pivot_column_index.reserve(columns_to_reduce.size());
//     // compute_pairs(columns_to_reduce, pivot_column_index, dim);
//     if (dim < dim_max){
//       assemble_columns_to_reduce(simplices, columns_to_reduce, pivot_column_index, dim + 1);
//     }
//   }
// }



template< SimplexIndexer Indexer > 
struct Cliqueser {
  const size_t n; 
  Indexer filter; 
  
  Cliqueser(const size_t _n, const size_t max_dim = 2) : n(_n), filter(_n) {
    BC.precompute(n, max_dim+2);
  }

  // Given simplex rank + its dimension, retrieve its vertices (store locally)
  // void simplex_vertices(const index_t simplex, const index_t dim, index_t* v_out) const {
  //   vertices[15] = simplex; 
  //   unrank_colex< false >(vertices.rbegin(), vertices.rbegin()+1, n, dim + 1, v_out);
  // }

  // Overloaded: given simplex rank + its dimension, retrieve its vertices (store locally)
  auto simplex_vertices(const index_t simplex, const index_t dim) const -> index_t* {
    vertices[15] = simplex; 
    unrank_colex< false >(vertices.rbegin(), vertices.rbegin()+1, n, dim + 1, vertices.begin());
    // simplex_vertices(simplex, dim, vertices.begin());
    return vertices.data();
  }

  // Given simplex rank + its dimension, retrieve its max weight
  auto simplex_weight(const index_t simplex, const index_t dim) const -> float {
    // vertices[15] = simplex; 
    // unrank_colex< false >(vertices.rbegin(), vertices.rbegin()+1, n, dim + 1, vertices.begin());
    index_t* sv = simplex_vertices(simplex, dim);
    return filter.simplex_index(sv, sv + dim + 1);
  }

  // Enumerates the ranks of the (dim+1)-cofacets on the coboundary of _simplex_rank
  template< typename Lambda > 
  void enum_coboundary(const index_t simplex, const index_t dim, Lambda&& f) const {
    // const auto R = CoboundaryRange(simplex, dim, n);
    // bool cont_enum = true;
    // for (auto c = R.begin(); c.has_next(); ++c){
    //   cont_enum = f(*c);
    // };
    index_t idx_below = simplex;
		index_t idx_above = 0;
		index_t j = n - 1;
		index_t k = dim + 1;
    bool cont_enum = true;
    while (j >= k && cont_enum){
      // std::cout << "j = " << j << ", k = " << k << std::endl;
      while ((static_cast< index_t >(BC.at(j,k)) <= idx_below)) {
        idx_below -= BC.at(j, k);
        idx_above += BC.at(j, k + 1);
        --j;
        --k;
        //assert(k != -1);
      }
      index_t cofacet_index = idx_above + BC.at(j--, k + 1) + idx_below;
      cont_enum = f(cofacet_index);
    }
  }

  // Given the rank of a (dim)-simplex 'simplex', enumerate the ranks of simplices on its boundary
  template< typename Lambda > 
  void enum_boundary(const index_t simplex, const index_t dim, Lambda&& f) const {
    index_t idx_below = simplex;
    index_t idx_above = 0; 
    index_t j = n - 1;
    bool cont_enum = true; 
    for (index_t k = dim; k >= 0 && cont_enum; --k){
      j = combinatorial::get_max_vertex< false >(idx_below, k + 1, j) - 1; // NOTE: Danger!
      index_t c = BC.at(j, k + 1);
      index_t face_index = idx_above - c + idx_below;
      idx_below -= c;
      idx_above += BC.at(j, k);
      cont_enum = f(face_index);
    }
  }
  
  // template< bool collect_edges = true > 
  // void compute_H0(
  //   const float threshold, 
  //   std::vector< IndexPair< float > >& edges,
  //   std::vector< IndexPair< float > >& pos_edges
  // ){
  //   // First collect and sort all the edges <= the supplied threshold by weight / colex rank
  //   auto edges = std::vector< IndexPair< float > >(); 
  //   // edges.reserve(n * (n-1) / 2); 
  //   filter.p_simplices(1, threshold, [](index_t* b, index_t* e, float weight){
  //     auto r = combinatorial::rank_colex_k(b, 2);
  //     edges.push_back(std::make_pair(r, weight));
  //   });
  //   std::sort(edges.rbegin(), edges.rend(), GreaterPair< float >);
    
  //   // Sweep the edges to build the connected components
  //   DisjointSet ds(n);
  //   auto edge = std::array< index_t, 2 >{ 0, 0 };
  //   for (auto e : edges) {
  //     const index_t* edge = simplex_vertices(e, 1, edge.begin());
  //     index_t u = ds.find(edge[0]);
  //     index_t v = ds.find(edge[1]);
  //     if (u != v) {
  //       // The edge is joining two components <=> it's a pivot / negative pair
  //       ds._union(u, v);
  //     } else if (apparent_zero_cofacet(e, 1) == -1){
  //       // The edge is already in connected component <=> it's a positive pair 
  //       // If in addition it does not have an apparent zero cofacet, then we save it
  //       pos_edges.push_back(e);
  //     }
  //   }
  //   // if (dim_max > 0) std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());
  // }

  // Given a dim-dimensional simplex, find its lexicographically maximal cofacet with identical simplex weight
  auto zero_cofacet(index_t cns_rank, size_t dim) const -> index_t {
    const auto c_weight = simplex_weight(cns_rank, dim);
    index_t zero_cofacet = -1; 
    enum_coboundary(cns_rank, dim, [&](const index_t cofacet) -> bool {
      const auto cofacet_weight = simplex_weight(cofacet, dim+1);
      if (cofacet_weight == c_weight){
        zero_cofacet = cofacet;
        return false; // end enum shortcut 
      }
      return true; 
    });
    return zero_cofacet; 
  }

  auto lex_minimal_facet(index_t cns_rank, size_t dim) const -> index_t {
    auto j = combinatorial::get_max_vertex< false >(cns_rank, dim + 1, n - 1) - 1; // danger! 
    return BC.at(j, dim + 1) + cns_rank;
  }

  auto lex_maximal_cofacet(index_t cns_rank, size_t dim) const -> index_t {
    index_t idx_below = cns_rank;
		index_t idx_above = 0;
		index_t j = n - 1;
		index_t k = dim + 1;
    while (j >= k){
      while ((static_cast< index_t >(BC.at(j,k)) <= idx_below)) {
        idx_below -= BC.at(j, k);
        idx_above += BC.at(j, k + 1);
        --j;
        --k;
        assert(k != -1);
      }
      index_t cofacet_index = idx_above + BC.at(j--, k + 1) + idx_below;
      return cofacet_index;
    }
    return -1; 
  }

  // Given a dim-dimensional simplex, find its lexicographically minimal facet with identical simplex weight
  auto zero_facet(index_t cns_rank, size_t dim) const -> index_t {
    const auto c_weight = simplex_weight(cns_rank, dim);
    index_t zero_facet = -1; 
    enum_boundary(cns_rank, dim, [&](const index_t facet) -> bool {
      const auto facet_weight = simplex_weight(facet, dim-1);
      if (facet_weight == c_weight){
        zero_facet = facet;
        return false; 
      }
      return true; 
    });
    return zero_facet; 
  }

  // Given a p-simplex, find it's (p-1)-face with identical filter weight and detect whether 
  // the two form a [reverse colexicographical] apparent pair in the simplexwise filtration
  auto apparent_zero_facet(const index_t cns_rank, const size_t p) const -> index_t {
    const auto facet = zero_facet(cns_rank, p);
    return (facet != -1) && (cns_rank == zero_cofacet(facet, p-1)) ? facet : -1; 
  }

  // Given a p-simplex, find it's (p+1)-face with identical filter weight and detect whether 
  // the two form a [reverse colexicographical] apparent pair in the simplexwise filtration
  auto apparent_zero_cofacet(const index_t cns_rank, const size_t p) const -> index_t {
    const auto cofacet = zero_cofacet(cns_rank, p);
    return (cofacet != -1) && (cns_rank == zero_facet(cofacet, p+1)) ? cofacet : -1; 
  }

  private: 
    mutable std::array< index_t, 16 > vertices; 
    // mutable BinomialCoefficientTable< 0, 0, index_t > BC; 
};

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

template< SimplexIndexer Indexer, typename Lambda >
void _clique_wrapper(py::module& m, std::string suffix, Lambda&& init){
  using F = typename Indexer::value_type; 
  using FT = Cliqueser< Indexer >;
  py::class_< FT >(m, (std::string("Cliqueser") + suffix).c_str())
		.def(py::init([](const index_t n, const int dim){
      return std::make_unique< FT >(FT(n, dim));
		}))
    .def("init", init)
    .def_readonly("n_vertices", &FT::n)
    .def("simplex_weight", &FT::simplex_weight)
    .def("boundary", [](const FT& M, const index_t cns_rank, const size_t dim) -> py_array< index_t > {
      auto facet_ranks = std::vector< index_t >();
      facet_ranks.reserve(dim+1);
      M.enum_boundary(cns_rank, dim, [&facet_ranks](const index_t facet){
        facet_ranks.push_back(facet);
        return true; 
      });
      return py::cast(facet_ranks);
    })
    .def("coboundary", [](const FT& M, const index_t cns_rank, const size_t dim) -> py_array< index_t > {
      auto cofacet_ranks = std::vector< index_t >();
      cofacet_ranks.reserve(M.n);
      M.enum_coboundary(cns_rank, dim, [&cofacet_ranks](const index_t cofacet){
        cofacet_ranks.push_back(cofacet);
        return true; 
      });
      return py::cast(cofacet_ranks);
    })
    .def("coboundary2", [](const FT& M, const index_t cns_rank, const size_t dim) -> py_array< index_t >{
      auto cofacet_ranks = std::vector< index_t >();
      cofacet_ranks.reserve(M.n);
      auto g = CoboundaryGenerator(cns_rank, dim, M.n);
      while (g.has_next()){
        cofacet_ranks.push_back(g.next());
      }
      return py::cast(cofacet_ranks);
    })
    .def("get_max_vertex", [](const FT& M, const size_t cns_rank, const size_t m) -> size_t {
      // Binary searches for the value K satisfying choose(K-1, m) <= r < choose(K, m) 
      return static_cast< size_t >(combinatorial::get_max_vertex< true >(cns_rank, m, M.n));
    })
    .def("lex_minimal_facet", &FT::lex_minimal_facet)
    .def("lex_maximal_cofacet", &FT::lex_maximal_cofacet)
    .def("zero_facet", &FT::zero_facet)
    .def("zero_cofacet", &FT::zero_cofacet)
    .def("apparent_zero_facet", &FT::apparent_zero_facet)
    .def("apparent_zero_cofacet", &FT::apparent_zero_cofacet)
    .def("apparent_zero_cofacets", [](const FT& M, const py_array< index_t >& cns_ranks, const size_t dim) { // -> py_array< index_t >
      const size_t nr = static_cast< const size_t >(cns_ranks.size());
      auto ap = std::vector< index_t >(nr);
      const index_t* r = cns_ranks.data(); 
      for (size_t i = 0; i < nr; ++i){
        ap[i] = M.apparent_zero_cofacet(r[i], dim);
      }
      auto ap_out = py_array< index_t >(nr, ap.data());
      return ap_out; 
    })
    .def("collect_cofacets", [](
      const FT& M, const py_array< index_t >& cns_ranks, const size_t dim, const float threshold, 
      const bool discard_pos = false, 
      const bool all_cofacets = false
    ) -> py_array< index_t > {
      auto cofacet_ranks = std::vector< index_t >();
      cofacet_ranks.reserve(cns_ranks.size());
      const index_t* R = cns_ranks.data();
      const auto append_cofacet = [&](const index_t cofacet){
        if (M.simplex_weight(cofacet, dim-1) <= threshold){ cofacet_ranks.push_back(cofacet); }
        return true; 
      };
      // std::for_each_n(cns_ranks.data(), cns_ranks.size())
      if (all_cofacets){
        for (size_t i = 0; i < size_t(cns_ranks.size()); ++i){
          enum_coboundary< true >(R[i], dim, M.n, append_cofacet);
        }
      } else {
        for (size_t i = 0; i < size_t(cns_ranks.size()); ++i){
          enum_coboundary< false >(R[i], dim, M.n, append_cofacet);
        }
      }
      return py::cast(cofacet_ranks); 
    })
    .def("p_simplices", [](const FT& M, const size_t p, const float threshold){
      auto p_simplices = std::vector< index_t >();
      M.filter.p_simplices(p, threshold, [p, &p_simplices](auto b, [[maybe_unused]] auto e, [[maybe_unused]] const float weight){
        auto r = combinatorial::rank_colex_k(b, p+1); // assumes b in reverse order
        p_simplices.push_back(r);
      });
      const auto out = py_array< index_t >(p_simplices.size(), p_simplices.data());
      return out;
    })
    .def("p_simplices2", [](const FT& M, const size_t p, const float threshold) -> py_array< index_t >{
      if (p <= 0){
        auto p_simplices = std::vector< index_t >(M.n);
        std::iota(p_simplices.rbegin(), p_simplices.rend(), 0);
        return py::cast(p_simplices);
      } else {
        const size_t NP = BC.at(M.n, p);
        const size_t NQ = BC.at(M.n, p+1);
        auto p_simplices = std::vector< index_t >(NQ);
        size_t ii = 0;
        for (auto r = 0; r < NP; ++r){
          enum_coboundary< false >(r, p-1, M.n, [&p_simplices, &ii](index_t cofacet){
            p_simplices[ii++] = cofacet;
            return true; 
          });
        }
        return py::cast(p_simplices);
      }  
    })
    // Sequentially constructs 
    // .def("enum_", [](){
    //     M.filter.p_simplices(p-1, threshold, [p, &p_simplices](auto b, [[maybe_unused]] auto e, [[maybe_unused]] const float weight){
    //       auto r = combinatorial::rank_colex_k(b, p+1); // assumes b in reverse order
    //       p_simplices.push_back(r);
    //     });
    //     enum_coboundary< false >()
    // })
    .def("cofacets_merged", [](const FT& M, const py_array< index_t >& cns_ranks, const size_t dim) -> py_array< index_t >{
      auto gens = std::vector< CoboundaryGenerator >();
      const index_t* R = cns_ranks.data();
      for (auto i = 0; i < cns_ranks.size(); ++i){
        gens.push_back(CoboundaryGenerator(R[i], dim, M.n));
      }
      auto cofacets = std::vector< index_t >();
      merge_k_sorted(gens, [&cofacets](index_t cofacet){
        cofacets.push_back(cofacet);
      });
      return py::cast(cofacets);
    })
    // Constructs the -psimplices + weights in the filtration up to a given threshold, optionally checking to see if 
    // each simplex participates in an apparent pair as a positive or negative simplex
    .def("build", [](const FT& M, const size_t p, const float threshold, bool check_pos = false, bool check_neg = false, const bool filter_pos = false){
      auto p_simplices = std::vector< index_t >();
      auto p_weights = std::vector< float >();
      auto p_status = std::vector< int >();
      auto apparent_check = std::function< int(index_t)>();
      check_neg = check_neg & (p > 0);
      check_pos = check_pos & (p < BC.BT.size());// BC.BT.size() == max_dim + 1 
      if (check_neg && check_pos){
        apparent_check = [&](const index_t r){
          const auto fa_r = M.apparent_zero_facet(r, p); // check if it has apparent zero facet 
          if (fa_r != -1){
            return -fa_r; // r is negative
          }
          const auto ca_r = M.apparent_zero_cofacet(r, p); 
          return ca_r == -1 ? 0 : ca_r; // r is positive 
        };
      } else if (check_neg){
        apparent_check = [&](const index_t r){ 
          const auto fa_r = M.apparent_zero_facet(r, p); 
          return fa_r != -1 ? -fa_r : 0; // r is negative
        };
      } else if (check_pos){
        apparent_check = [&](const index_t r){
          const auto ca_r = M.apparent_zero_cofacet(r, p); 
          return ca_r != -1 ? ca_r : 0;  // r is positive
        };
      } else {
        apparent_check = []([[maybe_unused]] const index_t r){ return 0; };
      }
      M.filter.p_simplices(p, threshold, [p, filter_pos, &apparent_check, &p_simplices, &p_weights, &p_status](auto b, [[maybe_unused]] auto e, const float w){
        const auto r = combinatorial::rank_colex_k(b, p+1); // assumes b in reverse order
        const auto af = apparent_check(r);
        if (!filter_pos){
          p_simplices.push_back(r);
          p_weights.push_back(w);
          p_status.push_back(af);
        } else {
          if (af <= 0){
            p_simplices.push_back(r);
            p_weights.push_back(w);
            p_status.push_back(af);
          }
        }
      });
      py_array< index_t > ps_out_(p_simplices.size(), p_simplices.data());
      py_array< float > pw_out_(p_weights.size(), p_weights.data());
      py_array< int > st_out_(p_status.size(), p_status.data());
      return py::make_tuple(ps_out_, pw_out_, st_out_);
    })
    ;
}
      // TODO: for starting edge, enumerate all the non-apparent triangles <= threshold (checked locally)
      // then, for remaining edges, do the same, at each pair
      // Do a sorted merge between each pair to keep the working memory of cofacets as small as possible

PYBIND11_MODULE(_clique, m) {

  _clique_wrapper< StarFilter< float > >(m, "_star", [](Cliqueser< StarFilter< float > >& C, std::vector< float > v_weights){
    if (v_weights.size() != C.n){
      throw std::invalid_argument("Invalid set of weights given.");
    }
    C.filter.weights = std::move(v_weights);
  });

  _clique_wrapper< FlagFilter< float > >(m, "_flag", [](Cliqueser< FlagFilter< float > >& C, std::vector< float > distances){
    if (distances.size() != (C.n * (C.n - 1) / 2)){
      throw std::invalid_argument("Invalid set of weights given.");
    }
    C.filter.weights = std::move(distances);
  });

  _clique_wrapper< MetricFilter< float > >(m, "_metric", [](Cliqueser< MetricFilter< float > >& C, std::vector< float > point_cloud, const size_t d){
    if (point_cloud.size() != (C.n * d)){
      throw std::invalid_argument("Invalid point cloud given. Must be (n x d).");
    }
    C.filter.d = d;
    C.filter.points = std::move(point_cloud);
  });

  m.def("enum_coboundary", [](const index_t simplex, const index_t dim, const index_t n, bool all_cofacets = true){
    // const auto R = CoboundaryRange(simplex, dim, n);
    if (BC.BT.size() <= size_t(dim+2) || BC.BT.at(0).size() < size_t(n)){
      BC.precompute(n, dim+2);
    }
    auto c_out = std::vector< index_t >();
    if (all_cofacets){
      enum_coboundary< true >(simplex, dim, n, [&](const index_t cofacet){
        c_out.push_back(cofacet);
        return true; 
      });
    } else {
      enum_coboundary< false >(simplex, dim, n, [&](const index_t cofacet){
        c_out.push_back(cofacet);
        return true; 
      });
    }
    return py::cast(c_out);
  });
  m.def("compress_coo", [](
    const py_array< int >& p_inc, const py_array< int >& q_inc, 
    const py_array< int >& r_ind, 
    const py_array< int >& c_ind,
    const py_array< float >& data
  ) -> py::tuple {
    const int* p_inc_ptr = p_inc.data();
    const int* q_inc_ptr = q_inc.data();
    const int* r_ptr = r_ind.data();
    const int* c_ptr = c_ind.data();
    const float* d_ptr = data.data();
    const size_t np = std::accumulate(p_inc_ptr, p_inc_ptr + p_inc.size(), 0);
    const size_t nq = std::accumulate(q_inc_ptr, q_inc_ptr + q_inc.size(), 0);
    auto r_out = std::vector< int >();
    auto c_out = std::vector< int >();
    auto d_out = std::vector< float >();
    r_out.reserve(nq*3);
    c_out.reserve(nq*3);
    d_out.reserve(nq*3);
    const size_t n_entries = static_cast< const size_t >(data.size());
    for (size_t i = 0; i < n_entries; ++i){
      if (p_inc_ptr[r_ptr[i]] & q_inc_ptr[c_ptr[i]]){
        r_out.push_back(r_ptr[i]);
        c_out.push_back(c_ptr[i]);
        d_out.push_back(d_ptr[i]);
      }
    }
    py_array< int > r_out_(r_out.size(), r_out.data());
    py_array< int > c_out_(c_out.size(), c_out.data());
    py_array< float > d_out_(d_out.size(), d_out.data());
    return py::make_tuple(d_out_, py::make_tuple(r_out_, c_out_));
    // return py::make_tuple(py::cast(r_out), py::cast(c_out), py::cast(d_out));
  })
  .def("build_coo", [](const size_t n, const size_t p, const py_array< index_t >& p_simplices, const py_array< index_t >& f_simplices){
    const index_t* ps = p_simplices.data();
    const size_t np = static_cast< size_t >(p_simplices.size());
    auto ri = std::vector< index_t >(np * (p+1));
    auto ci = std::vector< index_t >(np * (p+1));
    auto di = std::vector< float >(np * (p+1));
    size_t ii = 0; 

    // Make sure we've precomputed enough binomial coefficients  
    if (BC.BT.size() < p + 2 || BC.BT.at(0).size() < n){
      BC.precompute(n, p+2);
    }
  
    // TODO: break thi sinto three loops and parallelize
    float s; 
    for (auto j = 0; j < p_simplices.size(); ++j){
      s = -1.0; 
      enum_boundary(n, ps[j], p, [&](index_t r){
        s *= -1; 
        ri[ii] = r;
        ci[ii] = j;
        di[ii] = s;
        ++ii;
        return true; 
      });
    } 

    // Now, map the face ranks -> {0, 1, ..., n - 1}
    size_t c = 0; 
    auto index_map = std::unordered_map< index_t, index_t >();
    const index_t* F = f_simplices.data();
    for (auto i = 0; i != f_simplices.size(); ++i) {
      auto s = F[i];
      if (!index_map.contains(s)){
        index_map.insert({ s, c });
        // if (c <= 3){ std::cout << *r << " --> " << c << std::endl; }
        c++;
      }
    }
    std::for_each_n(ri.begin(), ri.size(), [&index_map](auto& r){
      r = index_map[r];
    });
    // std::transform(ri.begin(), ri.end(), ri.begin(), [&index_map](auto r){ return index_map[r]; });
    py_array< index_t > r_out_(ri.size(), ri.data());
    py_array< index_t > c_out_(ci.size(), ci.data());
    py_array< float > d_out_(di.size(), di.data());
    return py::make_tuple(d_out_, py::make_tuple(r_out_, c_out_));
  })
  ;
}
