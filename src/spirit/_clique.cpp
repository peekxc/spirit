#include "combin/include/combinatorial.h"
#include <vector> 
#include <cinttypes> 
#include <iostream>

using index_t = std::int64_t;
using combinatorial::BinomialCoefficientTable;

struct MetricSpace {
  const size_t n; 
  std::vector< float > weights; 
  mutable BinomialCoefficientTable< 0, 0, index_t > BC; 

  // template< typename Iter > 
  MetricSpace(const size_t _n, const size_t max_dim = 2) : n(_n), BC(){
    BC.precompute(n, max_dim+1);
  }

  // Enumerates the ranks of the (dim+1)-cofacets on the coboundary of _simplex_rank
  template< typename Lambda > 
  void enum_coboundary(const index_t simplex, const index_t dim, const bool all_cofacets, Lambda&& f) {

    // std::vector< index_t > vertices;// vertices.resize(_dim + 1);
    index_t idx_below = simplex;
		index_t idx_above = 0;
		index_t j = n - 1;
		index_t k = dim + 1;

    // Enumeration part
    while (j >= k && (all_cofacets || BC.at(j, k) > idx_below)){
      std::cout << "j : " << j << ", k = " << k << std::endl;
      while ((BC.at(j, k) <= idx_below)) {
        idx_below -= BC.at(j, k);
        idx_above += BC.at(j, k + 1);
        --j;
        --k;
        assert(k != -1);
        if (k < 0 || j < 0 || idx_below < 0){
          std::cout << "messed up" << std::endl; 
          break;
        }
      }
      index_t cofacet_index = idx_above + BC.at(j--, k + 1) + idx_below;
      f(cofacet_index);
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
	m.def("coboundary", [](const index_t cns_rank, const size_t dim, const size_t n) -> py_array< index_t > {
    // auto dist = std::vector(5, 0.0);
    // dist.begin(), dist.end()
    auto M = MetricSpace(n, dim);
    auto cofacet_ranks = std::vector< index_t >();
    cofacet_ranks.reserve(n);
    M.enum_coboundary(cns_rank, dim, true, [&cofacet_ranks](const index_t cofacet){
      cofacet_ranks.push_back(cofacet);
    });
    return py::cast(cofacet_ranks);
  });

  m.def("get_max_vertex", [](const size_t cns_rank, const size_t m, const size_t n) -> size_t {
    // Binary searches for the value K satisfying choose(K-1, m) <= r < choose(K, m) 
    return static_cast< size_t >(combinatorial::get_max_vertex< true >(cns_rank, m, n));
  });
}



