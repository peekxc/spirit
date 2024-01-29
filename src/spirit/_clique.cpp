#include "combin/include/combinatorial.h"
#include <vector> 
#include <cinttypes> 
#include <iostream>

using index_t = std::int64_t;
using combinatorial::BinomialCoefficientTable;
using combinatorial::unrank_colex;




struct MetricSpace {
  const size_t n; 
  bool flag = true; 
  std::vector< float > weights; 
  
  mutable std::array< index_t, 16 > vertices; 
  mutable BinomialCoefficientTable< 0, 0, index_t > BC; 

  // template< typename Iter > 
  MetricSpace(const size_t _n, const size_t max_dim = 2) : n(_n), BC(){
    BC.precompute(n, max_dim+2);
    // std::cout << "BC k size: " << BC.BT.size() << std::endl;
    // std::cout << "BC n size: " << BC.BT[0].size() << std::endl;
  }

  // Distance computation
  auto dist(const index_t i, const index_t j) const -> float {
    return weights.at(combinatorial::rank_lex_2(i,j,n));
  }


  // The initializer takes in either a set n-wweights or a set of (n choose 2)-weights
  void init(std::vector< float > _weights){
    if (_weights.size() == n){
      flag = false;
      weights = std::move(_weights);
    } else {
      if (_weights.size() != size_t(n * (n - 1) / 2)){
        throw std::invalid_argument("Invalid set of weights given.");
      }
      flag = true; 
      weights = std::move(_weights);
    }
  }

  template< typename Iter >
  auto _simplex_weight(Iter b, const Iter e) const {
    float s_weight = -std::numeric_limits< float >::infinity();
    if (flag){
      combinatorial::for_each_combination(b, b+2, e, [this, &s_weight](auto b, auto e){
        s_weight = std::max(s_weight, dist(*b, *(b+1))); 
        return false; 
      });
    } else {
      for (; b != e; ++b){
        s_weight = std::max(s_weight, weights[*b]);
      }
    }
    return s_weight;
  }

  // Given simplex rank + its dimension, retrieve its max weight
  auto simplex_weight(const index_t simplex, const index_t dim) const -> float {
    vertices[15] = simplex; 
    unrank_colex(vertices.rbegin(), vertices.rbegin()+1, n, dim + 1, vertices.begin());
    return _simplex_weight(vertices.begin(), vertices.begin()+dim+1);
  }
    // if (flag){

    //   for (index_t i = 0; i < dim+1){
    //     for (index_t j = i+1; j < dim+1; ++j){
    //       s_weight = std::max(s_weight, dist(vertices[i], vertices[j]));
    //     }
    //   }
    // } else {
    //   for (index_t i = 0; i < ){
    //     s_weight = std::max(s_weight, dist(vertices[i], vertices[j]));
    //   }
    // }
  // }

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
};


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

PYBIND11_MODULE(_clique, m) {
  py::class_< MetricSpace >(m, "MetricSpace")
		.def(py::init([](const index_t n, const int dim){
			return std::make_unique< MetricSpace >(MetricSpace(n, dim));
		}))
    .def("init", [](MetricSpace& M, const py_array< float >& dists, const float threshold){
      std::vector< float > _dists(dists.data(), dists.data() + dists.size());
      M.weights = std::move(_dists);
      M.flag = true;
    })
    .def_readonly("n", &MetricSpace::n)
    .def_readonly("flag", &MetricSpace::flag)
    .def_readonly("weights", &MetricSpace::weights)
    .def("boundary", [](const MetricSpace& M, const index_t cns_rank, const size_t dim) -> py_array< index_t > {
      auto facet_ranks = std::vector< index_t >();
      facet_ranks.reserve(dim+1);
      M.enum_boundary(cns_rank, dim, [&facet_ranks](const index_t facet){
        facet_ranks.push_back(facet);
        return true; 
      });
      return py::cast(facet_ranks);
    })
    .def("coboundary", [](const MetricSpace& M, const index_t cns_rank, const size_t dim) -> py_array< index_t > {
      auto cofacet_ranks = std::vector< index_t >();
      cofacet_ranks.reserve(M.n);
      M.enum_coboundary(cns_rank, dim, [&cofacet_ranks](const index_t cofacet){
        cofacet_ranks.push_back(cofacet);
        return true; 
      });
      return py::cast(cofacet_ranks);
    })
    .def("get_max_vertex", [](const MetricSpace& M, const size_t cns_rank, const size_t m) -> size_t {
      // Binary searches for the value K satisfying choose(K-1, m) <= r < choose(K, m) 
      return static_cast< size_t >(combinatorial::get_max_vertex< true >(cns_rank, m, M.n));
    })
    .def("apparent_facet", [](const MetricSpace& M, const index_t cns_rank, const size_t dim){
      const auto c_weight = M.simplex_weight(cns_rank, dim);
      index_t apparent_facet = -1; 
      M.enum_boundary(cns_rank, dim, [&](const index_t facet) -> bool {
        const auto facet_weight = M.simplex_weight(facet, dim-1);
        if (facet_weight == c_weight){
          apparent_facet = facet;
          return false; 
        }
        return true; 
      });
      return apparent_facet; 
    })
    ;
}



