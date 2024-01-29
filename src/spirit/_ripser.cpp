#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>


#include "io.h"

template < typename DistanceMatrix > 
struct ripser {
	const DistanceMatrix dist;
	const index_t n, dim_max;
	const value_t threshold;
	const coefficient_t modulus;
	const binomial_coeff_table binomial_coeff;
	const std::vector< coefficient_t > multiplicative_inverse;
	mutable std::vector< diameter_entry_t > cofacet_entries;
	mutable std::vector< index_t > vertices;

	struct entry_hash {
		std::size_t operator()(const entry_t& e) const { return hash<index_t>()(::get_index(e)); }
	};

	struct equal_index {
		bool operator()(const entry_t& e, const entry_t& f) const {
			return ::get_index(e) == ::get_index(f);
		}
	};

	// Based on: https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison
	bool equal_diam(const value_t& d1, const value_t& d2) {
		static constexpr value_t EPS = 128 * std::numeric_limits< value_t >::epsilon();
		static constexpr value_t ABS_TH = std::numeric_limits< value_t >::min();
		if (d1 == d2) return true;
		auto diff = std::abs(d1-d2);
		auto norm = std::min((std::abs(d1) + std::abs(d2)), std::numeric_limits< value_t >::max());
		return diff < std::max(ABS_TH, EPS * norm);
	}

	typedef hash_map<entry_t, size_t, entry_hash, equal_index> entry_hash_map;

	ripser(DistanceMatrix&& _dist, index_t _dim_max, value_t _threshold, coefficient_t _modulus)
	    : dist(std::move(_dist)), 
				n(dist.size()),
	      dim_max(std::min(_dim_max, 
				index_t(dist.size() - 2))), 
				threshold(_threshold), 
				modulus(_modulus), 
				binomial_coeff(n, dim_max + 2),
	      multiplicative_inverse(multiplicative_inverse_vector(_modulus)
	) {}

	index_t get_max_vertex(const index_t idx, const index_t k, const index_t n) const {
		return get_max(n, k - 1, [&](index_t w) -> bool { return (binomial_coeff(w, k) <= idx); });
	}

	index_t get_edge_index(const index_t i, const index_t j) const {
		return binomial_coeff(i, 2) + j;
	}

	template <typename OutputIterator>
	OutputIterator get_simplex_vertices(
		index_t idx, const index_t dim, index_t n, OutputIterator out
	) const {
		--n;
		for (index_t k = dim + 1; k > 1; --k) {
			n = get_max_vertex(idx, k, n);
			*out++ = n;
			idx -= binomial_coeff(n, k);
		}
		*out = idx;
		return out;
	}

	value_t compute_diameter(const index_t index, const index_t dim) const {
		value_t diam = -std::numeric_limits<value_t>::infinity();

		vertices.resize(dim + 1);
		get_simplex_vertices(index, dim, dist.size(), vertices.rbegin());

		for (index_t i = 0; i <= dim; ++i)
			for (index_t j = 0; j < i; ++j) {
				diam = std::max(diam, dist(vertices[i], vertices[j]));
			}
		return diam;
	}

	class simplex_coboundary_enumerator;

	class simplex_boundary_enumerator {
	private:
		index_t idx_below, idx_above, j, k;
		diameter_entry_t simplex;
		index_t dim;
		const coefficient_t modulus;
		const binomial_coeff_table& binomial_coeff;
		const ripser& parent;

	public:
		simplex_boundary_enumerator(const diameter_entry_t _simplex, const index_t _dim,
		                            const ripser& _parent)
		    : idx_below(get_index(_simplex)), idx_above(0), j(_parent.n - 1), k(_dim),
		      simplex(_simplex), modulus(_parent.modulus), binomial_coeff(_parent.binomial_coeff),
		      parent(_parent) {}

		simplex_boundary_enumerator(const index_t _dim, const ripser& _parent)
		    : simplex_boundary_enumerator(-1, _dim, _parent) {}

		void set_simplex(const diameter_entry_t _simplex, const index_t _dim) {
			idx_below = get_index(_simplex);
			idx_above = 0;
			j = parent.n - 1;
			k = _dim;
			simplex = _simplex;
			dim = _dim;
		}

		bool has_next() { return (k >= 0); }

		diameter_entry_t next() {
			std::cout << "here 1" << std::endl; 
			j = parent.get_max_vertex(idx_below, k + 1, j) - 1;

			std::cout << "here 2: " << idx_above << ", " << binomial_coeff(j, k + 1) << ", " << idx_below << std::endl; 
			index_t face_index = idx_above - binomial_coeff(j, k + 1) + idx_below;

			std::cout << "here 3" << std::endl; 
			value_t face_diameter = parent.compute_diameter(face_index, dim - 1);

			coefficient_t face_coefficient =
			    (k & 1 ? -1 + modulus : 1) * get_coefficient(simplex) % modulus;

			std::cout << "here 4" << std::endl; 
			idx_below -= binomial_coeff(j, k + 1);
			idx_above += binomial_coeff(j, k);

			--k;

			return diameter_entry_t(face_diameter, face_index, face_coefficient);
		}
	};

	diameter_entry_t get_zero_pivot_facet(const diameter_entry_t simplex, const index_t dim) {
		// static simplex_boundary_enumerator facets(0, *this);
		simplex_boundary_enumerator facets(0, *this); // NOTE: re-entry requires keeping this local or re-factoring to pass as argument
		facets.set_simplex(simplex, dim);
		while (facets.has_next()) {
			diameter_entry_t facet = facets.next();
			if (get_diameter(facet) == get_diameter(simplex)) return facet;
		}
		return diameter_entry_t(-1);
	}

	diameter_entry_t get_zero_pivot_cofacet(const diameter_entry_t simplex, const index_t dim) {
		// static simplex_coboundary_enumerator cofacets(*this);
		simplex_coboundary_enumerator cofacets(*this); // NOTE: re-entry requires keeping this local or re-factoring to pass as argument
		cofacets.set_simplex(simplex, dim);
		// std::cout << "diam(i:" << ::get_index(simplex) << ") = " << get_diameter(simplex) << std::endl;
		while (cofacets.has_next()) {
			diameter_entry_t cofacet = cofacets.next();
			// std::cout << "i:" << ::get_index(cofacet) << " (diam:" <<  get_diameter(cofacet) << "), " << std::flush;
			// if (equal_diam(get_diameter(cofacet), get_diameter(simplex))){
			// 	return cofacet;
			// }
			if (get_diameter(cofacet) == get_diameter(simplex)) return cofacet;
		}
		return diameter_entry_t(-1);
	}

	diameter_entry_t get_zero_apparent_facet(const diameter_entry_t simplex, const index_t dim) {
		diameter_entry_t facet = get_zero_pivot_facet(simplex, dim);
		return ((get_index(facet) != -1) &&
		        (get_index(get_zero_pivot_cofacet(facet, dim - 1)) == get_index(simplex)))
		           ? facet
		           : diameter_entry_t(-1);
	}

	diameter_entry_t get_zero_apparent_cofacet(const diameter_entry_t simplex, const index_t dim) {
		diameter_entry_t cofacet = get_zero_pivot_cofacet(simplex, dim);
		return ((get_index(cofacet) != -1) &&
		        (get_index(get_zero_pivot_facet(cofacet, dim + 1)) == get_index(simplex)))
		           ? cofacet
		           : diameter_entry_t(-1);
	}

	bool is_in_zero_apparent_pair(const diameter_entry_t simplex, const index_t dim) {
		return (get_index(get_zero_apparent_cofacet(simplex, dim)) != -1) ||
		       (get_index(get_zero_apparent_facet(simplex, dim)) != -1);
	}

	auto get_edges() -> std::vector<diameter_index_t>;
};

template <> 
class ripser< compressed_lower_distance_matrix>::simplex_coboundary_enumerator {
	index_t idx_below, idx_above, j, k;
	std::vector<index_t> vertices;
	diameter_entry_t simplex;
	
	const coefficient_t modulus;
	const compressed_lower_distance_matrix& dist;
	const binomial_coeff_table& binomial_coeff;
	const ripser& parent;

public:
	simplex_coboundary_enumerator(const diameter_entry_t _simplex, const index_t _dim,
	                              const ripser& _parent)
	    : modulus(_parent.modulus), dist(_parent.dist),
	      binomial_coeff(_parent.binomial_coeff), parent(_parent) {
		if (get_index(_simplex) != -1)
			parent.get_simplex_vertices(get_index(_simplex), _dim, parent.n, vertices.rbegin());
	}

	simplex_coboundary_enumerator(const ripser& _parent) : modulus(_parent.modulus), dist(_parent.dist),
	binomial_coeff(_parent.binomial_coeff), parent(_parent) {}

	void set_simplex(const diameter_entry_t _simplex, const index_t _dim) {
		idx_below = get_index(_simplex);
		idx_above = 0;
		j = parent.n - 1;
		k = _dim + 1;
		simplex = _simplex;
		vertices.resize(_dim + 1);
		parent.get_simplex_vertices(get_index(_simplex), _dim, parent.n, vertices.rbegin());
	}

	bool has_next(bool all_cofacets = true) {
		return (j >= k && (all_cofacets || binomial_coeff(j, k) > idx_below));
	}

	diameter_entry_t next() {
		while ((binomial_coeff(j, k) <= idx_below)) {
			idx_below -= binomial_coeff(j, k);
			idx_above += binomial_coeff(j, k + 1);
			--j;
			--k;
			assert(k != -1);
		}
		value_t cofacet_diameter = get_diameter(simplex);
		for (index_t i : vertices) cofacet_diameter = std::max(cofacet_diameter, dist(j, i));
		index_t cofacet_index = idx_above + binomial_coeff(j--, k + 1) + idx_below;
		coefficient_t cofacet_coefficient =
		    (k & 1 ? modulus - 1 : 1) * get_coefficient(simplex) % modulus;
		return diameter_entry_t(cofacet_diameter, cofacet_index, cofacet_coefficient);
	}
};

template <> 
class ripser< sparse_distance_matrix >::simplex_coboundary_enumerator {
	index_t idx_below, idx_above, k;
	std::vector<index_t> vertices;
	diameter_entry_t simplex;
	const coefficient_t modulus;
	const sparse_distance_matrix& dist;
	const binomial_coeff_table& binomial_coeff;
	std::vector<std::vector<index_diameter_t>::const_reverse_iterator> neighbor_it;
	std::vector<std::vector<index_diameter_t>::const_reverse_iterator> neighbor_end;
	index_diameter_t neighbor;
	const ripser& parent;

public:
	simplex_coboundary_enumerator(const diameter_entry_t _simplex, const index_t _dim,
	                              const ripser& _parent)
	    : modulus(_parent.modulus), dist(_parent.dist),
	      binomial_coeff(_parent.binomial_coeff), parent(_parent) {
		if (get_index(_simplex) != -1) set_simplex(_simplex, _dim);
	}

	simplex_coboundary_enumerator(const ripser& _parent)
	    : modulus(_parent.modulus), dist(_parent.dist),
	binomial_coeff(_parent.binomial_coeff), parent(_parent) {}

	void set_simplex(const diameter_entry_t _simplex, const index_t _dim) {
		idx_below = get_index(_simplex);
		idx_above = 0;
		k = _dim + 1;
		simplex = _simplex;
		vertices.resize(_dim + 1);
		parent.get_simplex_vertices(idx_below, _dim, parent.n, vertices.rbegin());

		neighbor_it.resize(_dim + 1);
		neighbor_end.resize(_dim + 1);
		for (index_t i = 0; i <= _dim; ++i) {
			auto v = vertices[i];
			neighbor_it[i] = dist.neighbors[v].rbegin();
			neighbor_end[i] = dist.neighbors[v].rend();
		}
	}

	bool has_next(bool all_cofacets = true) {
		for (auto &it0 = neighbor_it[0], &end0 = neighbor_end[0]; it0 != end0; ++it0) {
			neighbor = *it0;
			for (size_t idx = 1; idx < neighbor_it.size(); ++idx) {
				auto &it = neighbor_it[idx], end = neighbor_end[idx];
				while (get_index(*it) > get_index(neighbor))
					if (++it == end) return false;
				if (get_index(*it) != get_index(neighbor))
					goto continue_outer;
				else
					neighbor = std::max(neighbor, *it);
			}
			while (k > 0 && vertices[k - 1] > get_index(neighbor)) {
				if (!all_cofacets) return false;
				idx_below -= binomial_coeff(vertices[k - 1], k);
				idx_above += binomial_coeff(vertices[k - 1], k + 1);
				--k;
			}
			return true;
		continue_outer:;
		}
		return false;
	}

	diameter_entry_t next() {
		++neighbor_it[0];
		value_t cofacet_diameter = std::max(get_diameter(simplex), get_diameter(neighbor));
		index_t cofacet_index = idx_above + binomial_coeff(get_index(neighbor), k + 1) + idx_below;
		coefficient_t cofacet_coefficient =
		    (k & 1 ? modulus - 1 : 1) * get_coefficient(simplex) % modulus;
		return diameter_entry_t(cofacet_diameter, cofacet_index, cofacet_coefficient);
	}
};

template <> 
auto ripser<compressed_lower_distance_matrix>::get_edges() -> std::vector<diameter_index_t> {
	std::vector<diameter_index_t> edges;
	std::vector<index_t> vertices(2);
	for (index_t index = binomial_coeff(n, 2); index-- > 0;) {
		get_simplex_vertices(index, 1, dist.size(), vertices.rbegin());
		value_t length = dist(vertices[0], vertices[1]);
		if (length <= threshold) edges.push_back({length, index});
	}
	return edges;
}

template <> 
auto ripser<sparse_distance_matrix>::get_edges() -> std::vector<diameter_index_t> {
	std::vector<diameter_index_t> edges;
	for (index_t i = 0; i < n; ++i)
		for (auto n : dist.neighbors[i]) {
			index_t j = get_index(n);
			if (i > j) edges.push_back({get_diameter(n), get_edge_index(i, j)});
		}
	return edges;
}


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

//storing the entries of the lower (or upper) triangular part of the distance matrix in a std::vector, 
// sorted lexicographically by row index, then column index
template< typename DistanceMatrix >
void _ripser_wrapper(py::module& m, std::string suffix){
	using RIP = ripser< DistanceMatrix >;
	py::class_< RIP >(m, (std::string("ripser") + suffix).c_str())
		.def(py::init([](const py_array< float >& dists, const int dim, const float threshold){
			std::vector< float > _dists(dists.data(), dists.data() + dists.size());
			return std::make_unique< ripser< DistanceMatrix > >(
				DistanceMatrix(std::move(_dists)), index_t(dim), value_t(threshold), coefficient_t(2)
			);
		}))
		.def_readonly("n", &RIP::n)
		.def_readonly("dim_max", &RIP::dim_max)
		.def_readonly("threshold", &RIP::threshold)
		.def_readonly("modulus", &RIP::modulus)
		.def_readonly("vertices", &RIP::vertices)
  	.def("_apparent_cofacet", [](RIP& RM, const index_t cns_rank, const float diam, const index_t dim) -> index_t {
			const auto simplex = diameter_entry_t(diam, cns_rank, coefficient_t(1));
			const auto cofacet_pair = RM.get_zero_apparent_cofacet(simplex, dim);
			return ::get_index(cofacet_pair); // dim == dim(cns_rank) - 1
		})
		.def("_apparent_facet", [](RIP& RM, const index_t cns_rank, const float diam, const index_t dim) -> index_t {
			const auto simplex = diameter_entry_t(diam, cns_rank, coefficient_t(1));
			const auto facet_pair = RM.get_zero_apparent_facet(simplex, dim);
			return ::get_index(facet_pair);
		})
		.def("apparent_facets", [](RIP& RM, const py_array< index_t >& cns_ranks, const py_array< float >& weights, const index_t dim) -> py_array< index_t > {
			// Given colex ranks of (dim)-simplices + their filtration weights, finds their (dim-1)-facets, if they exist
			assert(cns_ranks.size() == weights.size());
			assert(dim > 0);
			const index_t* ranks = cns_ranks.data();
			const float* sweight = weights.data();
			const size_t n_simplices = static_cast< size_t >(cns_ranks.size());
			
			auto facet_ranks = std::vector< index_t >(n_simplices, 0);
			for (size_t i = 0; i < n_simplices; ++i){
				const auto simplex = diameter_entry_t(sweight[i], ranks[i], coefficient_t(1));
				const auto facet_pair = RM.get_zero_apparent_cofacet(simplex, dim-1);
				facet_ranks[i] = ::get_index(facet_pair);
			}
			return py::cast(facet_ranks);
		})
		.def("apparent_cofacets", [](RIP& RM, const py_array< index_t >& cns_ranks, const py_array< float >& weights, const index_t dim) -> py_array< index_t > {
			// Given colex ranks of (dim)-simplices + their filtration weights, finds their (dim+1)-cofacets, if they exist
			assert(cns_ranks.size() == weights.size());
			assert(dim > 0);
			const index_t* ranks = cns_ranks.data();
			const float* sweight = weights.data();
			const size_t n_simplices = static_cast< size_t >(cns_ranks.size());
			
			auto cofacet_ranks = std::vector< index_t >(n_simplices, 0);
			for (size_t i = 0; i < n_simplices; ++i){
				const auto simplex = diameter_entry_t(sweight[i], ranks[i], coefficient_t(1));
				const auto cofacet_pair = RM.get_zero_apparent_facet(simplex, dim+1);
				cofacet_ranks[i] = ::get_index(cofacet_pair);
			}
			return py::cast(cofacet_ranks);
		})
		.def("diameters", [](RIP& RM, const py_array< index_t >& cns_ranks, const index_t dim) -> py_array< float >{
			const index_t* ranks = cns_ranks.data();
			const size_t n_simplices = static_cast< size_t >(cns_ranks.size());
			auto diams = std::vector< float >(n_simplices, 0.0);
			for (size_t i = 0; i < n_simplices; ++i){
				diams[i] = RM.compute_diameter(ranks[i], dim);
			}
			return py::cast(diams);
		})
		;
}

PYBIND11_MODULE(_ripser, m) {
	_ripser_wrapper< compressed_lower_distance_matrix >(m, "_lower");
	// _ripser_wrapper< compressed_upper_distance_matrix >(m, "_upper");
	// _ripser_wrapper< euclidean_distance_matrix >(m, "_euclidean");
	// _ripser_wrapper< compressed_distance_matrix< POINT_CLOUD > >(m, "_pc");
}


// enum file_format {
// 	LOWER_DISTANCE_MATRIX,
// 	UPPER_DISTANCE_MATRIX,
// 	DISTANCE_MATRIX,
// 	POINT_CLOUD,
// 	DIPHA,
// 	SPARSE,
// 	BINARY
// };

// static const uint16_t endian_check(0xff00);
// static const bool is_big_endian = *reinterpret_cast<const uint8_t*>(&endian_check);

// template <typename T> T read(std::istream& input_stream) {
// 	T result;
// 	char* p = reinterpret_cast<char*>(&result);
// 	if (input_stream.read(p, sizeof(T)).gcount() != sizeof(T)) return T();
// 	if (is_big_endian) std::reverse(p, p + sizeof(T));
// 	return result;
// }

// euclidean_distance_matrix read_point_cloud(std::istream& input_stream) {
// 	std::vector<std::vector<value_t>> points;

// 	std::string line;
// 	value_t value;
// 	while (std::getline(input_stream, line)) {
// 		std::vector<value_t> point;
// 		std::istringstream s(line);
// 		while (s >> value) {
// 			point.push_back(value);
// 			s.ignore();
// 		}
// 		if (!point.empty()) points.push_back(point);
// 		assert(point.size() == points.front().size());
// 	}

// 	euclidean_distance_matrix eucl_dist(std::move(points));
// 	index_t n = eucl_dist.size();
// 	std::cout << "point cloud with " << n << " points in dimension "
// 	          << eucl_dist.points.front().size() << std::endl;

// 	return eucl_dist;
// }

// sparse_distance_matrix read_sparse_distance_matrix(std::istream& input_stream) {
// 	std::vector<std::vector<index_diameter_t>> neighbors;
// 	index_t num_edges = 0;

// 	std::string line;
// 	while (std::getline(input_stream, line)) {
// 		std::istringstream s(line);
// 		size_t i, j;
// 		value_t value;
// 		s >> i;
// 		s.ignore();
// 		s >> j;
// 		s.ignore();
// 		s >> value;
// 		s.ignore();
// 		if (i != j) {
// 			neighbors.resize(std::max({neighbors.size(), i + 1, j + 1}));
// 			neighbors[i].push_back({j, value});
// 			neighbors[j].push_back({i, value});
// 			++num_edges;
// 		}
// 	}

// 	for (size_t i = 0; i < neighbors.size(); ++i)
// 		std::sort(neighbors[i].begin(), neighbors[i].end());

// 	return sparse_distance_matrix(std::move(neighbors), num_edges);
// }

// compressed_lower_distance_matrix read_lower_distance_matrix(std::istream& input_stream) {
// 	std::vector<value_t> distances;
// 	value_t value;
// 	while (input_stream >> value) {
// 		distances.push_back(value);
// 		input_stream.ignore();
// 	}

// 	return compressed_lower_distance_matrix(std::move(distances));
// }

// compressed_lower_distance_matrix read_upper_distance_matrix(std::istream& input_stream) {
// 	std::vector<value_t> distances;
// 	value_t value;
// 	while (input_stream >> value) {
// 		distances.push_back(value);
// 		input_stream.ignore();
// 	}

// 	return compressed_lower_distance_matrix(compressed_upper_distance_matrix(std::move(distances)));
// }

// compressed_lower_distance_matrix read_distance_matrix(std::istream& input_stream) {
// 	std::vector<value_t> distances;

// 	std::string line;
// 	value_t value;
// 	for (int i = 0; std::getline(input_stream, line); ++i) {
// 		std::istringstream s(line);
// 		for (int j = 0; j < i && s >> value; ++j) {
// 			distances.push_back(value);
// 			s.ignore();
// 		}
// 	}

// 	return compressed_lower_distance_matrix(std::move(distances));
// }

// compressed_lower_distance_matrix read_dipha(std::istream& input_stream) {
// 	if (read<int64_t>(input_stream) != 8067171840) {
// 		std::cerr << "input is not a Dipha file (magic number: 8067171840)" << std::endl;
// 		exit(-1);
// 	}

// 	if (read<int64_t>(input_stream) != 7) {
// 		std::cerr << "input is not a Dipha distance matrix (file type: 7)" << std::endl;
// 		exit(-1);
// 	}

// 	index_t n = read<int64_t>(input_stream);

// 	std::vector<value_t> distances;

// 	for (int i = 0; i < n; ++i)
// 		for (int j = 0; j < n; ++j)
// 			if (i > j)
// 				distances.push_back(read<double>(input_stream));
// 			else
// 				read<double>(input_stream);

// 	return compressed_lower_distance_matrix(std::move(distances));
// }

// compressed_lower_distance_matrix read_binary(std::istream& input_stream) {
// 	std::vector<value_t> distances;
// 	while (!input_stream.eof()) distances.push_back(read<value_t>(input_stream));
// 	return compressed_lower_distance_matrix(std::move(distances));
// }

// compressed_lower_distance_matrix read_file(std::istream& input_stream, const file_format format) {
// 	switch (format) {
// 	case LOWER_DISTANCE_MATRIX:
// 		return read_lower_distance_matrix(input_stream);
// 	case UPPER_DISTANCE_MATRIX:
// 		return read_upper_distance_matrix(input_stream);
// 	case DISTANCE_MATRIX:
// 		return read_distance_matrix(input_stream);
// 	case POINT_CLOUD:
// 		return read_point_cloud(input_stream);
// 	case DIPHA:
// 		return read_dipha(input_stream);
// 	default:
// 		return read_binary(input_stream);
// 	}
// }

// void print_usage_and_exit(int exit_code) {
// 	std::cerr
// 	    << "Usage: "
// 	    << "ripser "
// 	    << "[options] [filename]" << std::endl
// 	    << std::endl
// 	    << "Options:" << std::endl
// 	    << std::endl
// 	    << "  --help           print this screen" << std::endl
// 	    << "  --format         use the specified file format for the input. Options are:"
// 	    << std::endl
// 	    << "                     lower-distance (lower triangular distance matrix)"
// 	    << std::endl
// 	    << "                     upper-distance (upper triangular distance matrix)" << std::endl
// 	    << "         (default:)  distance       (distance matrix; only lower triangular part is read)" << std::endl
// 	    << "                     point-cloud    (point cloud in Euclidean space)" << std::endl
// 	    << "                     dipha          (distance matrix in DIPHA file format)" << std::endl
// 	    << "                     sparse         (sparse distance matrix in sparse triplet format)"
// 	    << std::endl
// 	    << "                     binary         (lower triangular distance matrix in binary format)"
// 	    << std::endl
// 	    << "  --dim <k>        compute persistent homology up to dimension k" << std::endl
// 	    << "  --threshold <t>  compute Rips complexes up to diameter t" << std::endl
// #ifdef USE_COEFFICIENTS
// 	    << "  --modulus <p>    compute homology with coefficients in the prime field Z/pZ"
// 	    << std::endl
// #endif
// 	    << "  --ratio <r>      only show persistence pairs with death/birth ratio > r" << std::endl
// 	    << std::endl;
// 	exit(exit_code);
// }

// int main(int argc, char** argv) {
// 	const char* filename = nullptr;

// 	file_format format = DISTANCE_MATRIX;

// 	index_t dim_max = 1;
// 	value_t threshold = std::numeric_limits<value_t>::max();
// 	float ratio = 1;
// 	coefficient_t modulus = 2;

// 	for (index_t i = 1; i < argc; ++i) {
// 		const std::string arg(argv[i]);
// 		if (arg == "--help") {
// 			print_usage_and_exit(0);
// 		} else if (arg == "--dim") {
// 			std::string parameter = std::string(argv[++i]);
// 			size_t next_pos;
// 			dim_max = std::stol(parameter, &next_pos);
// 			if (next_pos != parameter.size()) print_usage_and_exit(-1);
// 		} else if (arg == "--threshold") {
// 			std::string parameter = std::string(argv[++i]);
// 			size_t next_pos;
// 			threshold = std::stof(parameter, &next_pos);
// 			if (next_pos != parameter.size()) print_usage_and_exit(-1);
// 		} else if (arg == "--ratio") {
// 			std::string parameter = std::string(argv[++i]);
// 			size_t next_pos;
// 			ratio = std::stof(parameter, &next_pos);
// 			if (next_pos != parameter.size()) print_usage_and_exit(-1);
// 		} else if (arg == "--format") {
// 			std::string parameter = std::string(argv[++i]);
// 			if (parameter.rfind("lower", 0) == 0)
// 				format = LOWER_DISTANCE_MATRIX;
// 			else if (parameter.rfind("upper", 0) == 0)
// 				format = UPPER_DISTANCE_MATRIX;
// 			else if (parameter.rfind("dist", 0) == 0)
// 				format = DISTANCE_MATRIX;
// 			else if (parameter.rfind("point", 0) == 0)
// 				format = POINT_CLOUD;
// 			else if (parameter == "dipha")
// 				format = DIPHA;
// 			else if (parameter == "sparse")
// 				format = SPARSE;
// 			else if (parameter == "binary")
// 				format = BINARY;
// 			else
// 				print_usage_and_exit(-1);
// #ifdef USE_COEFFICIENTS
// 		} else if (arg == "--modulus") {
// 			std::string parameter = std::string(argv[++i]);
// 			size_t next_pos;
// 			modulus = std::stol(parameter, &next_pos);
// 			if (next_pos != parameter.size() || !is_prime(modulus)) print_usage_and_exit(-1);
// #endif
// 		} else {
// 			if (filename) { print_usage_and_exit(-1); }
// 			filename = argv[i];
// 		}
// 	}

// 	std::ifstream file_stream(filename);
// 	if (filename && file_stream.fail()) {
// 		std::cerr << "couldn't open file " << filename << std::endl;
// 		exit(-1);
// 	}

// 	if (format == SPARSE) {
// 		sparse_distance_matrix dist =
// 		    read_sparse_distance_matrix(filename ? file_stream : std::cin);
// 		std::cout << "sparse distance matrix with " << dist.size() << " points and "
// 		          << dist.num_edges << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
// 		          << std::endl;

// 		ripser<sparse_distance_matrix>(std::move(dist), dim_max, threshold, ratio, modulus)
// 		    .compute_barcodes();
// 	} else if (format == POINT_CLOUD && threshold < std::numeric_limits<value_t>::max()) {
// 		sparse_distance_matrix dist(read_point_cloud(filename ? file_stream : std::cin), threshold);
// 		ripser<sparse_distance_matrix>(std::move(dist), dim_max, threshold, ratio, modulus)
// 				.compute_barcodes();
// 	} else {
// 		compressed_lower_distance_matrix dist =
// 		    read_file(filename ? file_stream : std::cin, format);

// 		value_t min = std::numeric_limits<value_t>::infinity(),
// 		        max = -std::numeric_limits<value_t>::infinity(), max_finite = max;
// 		int num_edges = 0;

// 		value_t enclosing_radius = std::numeric_limits<value_t>::infinity();
// 		if (threshold == std::numeric_limits<value_t>::max()) {
// 			for (size_t i = 0; i < dist.size(); ++i) {
// 				value_t r_i = -std::numeric_limits<value_t>::infinity();
// 				for (size_t j = 0; j < dist.size(); ++j) r_i = std::max(r_i, dist(i, j));
// 				enclosing_radius = std::min(enclosing_radius, r_i);
// 			}
// 		}

// 		for (auto d : dist.distances) {
// 			min = std::min(min, d);
// 			max = std::max(max, d);
// 			if (d != std::numeric_limits<value_t>::infinity()) max_finite = std::max(max_finite, d);
// 			if (d <= threshold) ++num_edges;
// 		}
// 		std::cout << "value range: [" << min << "," << max_finite << "]" << std::endl;

// 		if (threshold == std::numeric_limits<value_t>::max()) {
// 			std::cout << "distance matrix with " << dist.size()
// 			          << " points, using threshold at enclosing radius " << enclosing_radius
// 			          << std::endl;
// 			ripser<compressed_lower_distance_matrix>(std::move(dist), dim_max, enclosing_radius,
// 			                                         ratio, modulus)
// 			    .compute_barcodes();
// 		} else {
// 			std::cout << "sparse distance matrix with " << dist.size() << " points and "
// 			          << num_edges << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
// 			          << std::endl;

// 			ripser<sparse_distance_matrix>(sparse_distance_matrix(std::move(dist), threshold),
// 			                               dim_max, threshold, ratio, modulus)
// 			    .compute_barcodes();
// 		}
// 		exit(0);
// 	}
// }