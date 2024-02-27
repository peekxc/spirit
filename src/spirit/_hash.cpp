#include <vector> 
#include <cinttypes> 
#include <iostream>
#include <iterator>
#include <concepts> 
#include <type_traits>
#include <numeric>
#include <cinttypes>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <emhash5.h>
#include <pthash/pthash.hpp>

// Benchmarking hashing implementations (+ including minimal perfect ones!)
PYBIND11_MODULE(_hash, m) {
  m.def("benchmark_pthash", [](const std::vector< uint_fast64_t >& t_ranks, const std::vector< uint_fast64_t >& e_ranks){
    

  })
}
