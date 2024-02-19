#include <vector> 
#include <cinttypes> 
#include <iostream>
#include <iterator>
#include <concepts> 
#include <type_traits>
#include <numeric>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <combin/combinatorial.h>

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

  template< std::integral I >
  auto simplex_index(I* b, I* e) const -> T {
    T s_weight = -std::numeric_limits< T >::infinity(); 
    for (; b != e; ++b){
      s_weight = std::max(s_weight, weights[*b]);
    }
    return s_weight; 
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

  template< std::integral I >
  auto simplex_index(I* b, I* e) const -> T {
    T s_weight = -std::numeric_limits< T >::infinity();
    combinatorial::for_each_combination(b, b+2, e, [this, &s_weight](auto b, [[maybe_unused]] auto e){
      s_weight = std::max(
        s_weight, 
        weights.at(combinatorial::rank_lex_2(*b, *(b+1), n))
      ); 
      return false; 
    });
    return s_weight;
  }

  // Given index of simplex S, its weight, and the vertex id 'v' of face F = S - {v}
  // auto facet_weight(){

  // }

};


template< typename T >
struct MetricFilter {
  using value_type = T;
  
  const size_t n; 
  size_t d; 
  std::vector< T > points; 

  MetricFilter(const size_t n_) : n(n_) {}
  MetricFilter(const size_t n_, const size_t d_, std::vector< T > coords) : n(n_), d(d_), points(coords) {
  };

  template< std::integral I >
  auto simplex_index(I* b, I* e) const -> T {
    T s_weight = -std::numeric_limits< T >::infinity();
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

  // Given index of simplex S, its weight, and the vertex id 'v' of face F = S - {v}
  // auto facet_weight(){

  // }

};




template< SimplexIndexer Indexer > 
struct Cliqueser {
  const size_t n; 
  Indexer filter; 

  Cliqueser(const size_t _n, const size_t max_dim = 2) : n(_n), filter(_n), BC() {
    BC.precompute(n, max_dim+2);
  }

  // Given simplex rank + its dimension, retrieve its max weight
  auto simplex_weight(const index_t simplex, const index_t dim) const -> float {
    vertices[15] = simplex; 
    unrank_colex(vertices.rbegin(), vertices.rbegin()+1, n, dim + 1, vertices.begin());
    return filter.simplex_index(vertices.data(), vertices.data() + dim + 1);
  }

  // Enumerates the ranks of the (dim+1)-cofacets on the coboundary of _simplex_rank
  template< typename Lambda > 
  void enum_coboundary(const index_t simplex, const index_t dim, Lambda&& f) const {
    index_t idx_below = simplex;
		index_t idx_above = 0;
		index_t j = n - 1;
		index_t k = dim + 1;
    bool cont_enum = true;
    while (j >= k && cont_enum){
      // std::cout << "j = " << j << ", k = " << k << std::endl;
      while ((BC.at(j,k) <= idx_below)) {
        idx_below -= BC.at(j, k);
        idx_above += BC.at(j, k + 1);
        --j;
        --k;
        assert(k != -1);
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
      j = combinatorial::get_max_vertex< true >(idx_below, k + 1, j) - 1;
      index_t c = BC.at(j, k + 1);
      index_t face_index = idx_above - c + idx_below;
      idx_below -= c;
      idx_above += BC.at(j, k);
      cont_enum = f(face_index);
    }
  }

  // Given a dim-dimensional simplex, find its lexicographically maximal cofacet with identical simplex weight
  auto zero_cofacet(index_t cns_rank, size_t dim) const -> index_t {
    const auto c_weight = simplex_weight(cns_rank, dim);
    index_t zero_cofacet = -1; 
    enum_coboundary(cns_rank, dim, [&](const index_t cofacet) -> bool {
      const auto cofacet_weight = simplex_weight(cofacet, dim+1);
      if (cofacet_weight == c_weight){
        zero_cofacet = cofacet;
        return false; 
      }
      return true; 
    });
    return zero_cofacet; 
  }

  auto lex_minimal_facet(index_t cns_rank, size_t dim) const -> index_t {
    auto j = combinatorial::get_max_vertex< true >(cns_rank, dim + 1, n - 1) - 1;
    return BC.at(j, dim + 1) + cns_rank;
  }

  auto lex_maximal_cofacet(index_t cns_rank, size_t dim) const -> index_t {
    index_t idx_below = cns_rank;
		index_t idx_above = 0;
		index_t j = n - 1;
		index_t k = dim + 1;
    while (j >= k){
      while ((BC.at(j,k) <= idx_below)) {
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

  // Given a (dim)-dimensional simplex, find it's facet with identical simplex weight and detect whether 
  // the two form an apparent pair in the reverse colexicographically-refined filtration
  auto apparent_zero_facet(index_t cns_rank, size_t dim) const -> index_t {
    const auto facet = zero_facet(cns_rank, dim);
    return (facet != -1) && (cns_rank == zero_cofacet(facet, dim-1)) ? facet : -1; 
  }

  // Given a (dim)-dimensional simplex, find it's cofacet with identical simplex weight and detect whether 
  // the two form an apparent pair in the reverse colexicographically-refined filtration
  auto apparent_zero_cofacet(index_t cns_rank, size_t dim) const -> index_t {
    const auto cofacet = zero_cofacet(cns_rank, dim);
    return (cofacet != -1) && (cns_rank == zero_facet(cofacet, dim+1)) ? cofacet : -1; 
  }

  private: 
    mutable std::array< index_t, 16 > vertices; 
    mutable BinomialCoefficientTable< 0, 0, index_t > BC; 
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
    ;
}

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
}
