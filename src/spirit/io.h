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

template <class Key, class T, class H, class E> using hash_map = std::unordered_map<Key, T, H, E>;
template <class Key> using hash = std::hash<Key>;

typedef float value_t;
typedef int64_t index_t;
typedef uint16_t coefficient_t;


static const std::string clear_line("\r\033[K");

static const size_t num_coefficient_bits = 8;

static const index_t max_simplex_index = (index_t(1) << (8 * sizeof(index_t) - 1 - num_coefficient_bits)) - 1;

void check_overflow(index_t i) {
	if
#ifdef USE_COEFFICIENTS
	    (i > max_simplex_index)
#else
	    (i < 0)
#endif
		throw std::overflow_error("simplex index " + std::to_string((uint64_t)i) +
		                          " in filtration is larger than maximum index " +
		                          std::to_string(max_simplex_index));
}

struct binomial_coeff_table {
	std::vector< std::vector< index_t> > B;
	
  binomial_coeff_table(index_t n, index_t k) : B(k + 1, std::vector<index_t>(n + 1, 0)) {
		for (index_t i = 0; i <= n; ++i) {
			B[0][i] = 1;
			for (index_t j = 1; j < std::min(i, k + 1); ++j)
				B[j][i] = B[j - 1][i - 1] + B[j][i - 1];
			if (i <= k) B[i][i] = 1;
			check_overflow(B[std::min(i >> 1, k)][i]);
		}
	}

	index_t operator()(index_t n, index_t k) const {
		assert(n < B.size() && k < B[n].size() && n >= k - 1);
		return B[k][n];
	}
};

bool is_prime(const coefficient_t n) {
	if (!(n & 1) || n < 2) return n == 2;
	for (coefficient_t p = 3; p <= n / p; p += 2)
		if (!(n % p)) return false;
	return true;
}

std::vector<coefficient_t> multiplicative_inverse_vector(const coefficient_t m) {
	std::vector<coefficient_t> inverse(m);
	inverse[1] = 1;
	// m = a * (m / a) + m % a
	// Multipying with inverse(a) * inverse(m % a):
	// 0 = inverse(m % a) * (m / a) + inverse(a)  (mod m)
	for (coefficient_t a = 2; a < m; ++a) inverse[a] = m - (inverse[m % a] * (m / a)) % m;
	return inverse;
}

typedef index_t entry_t;
index_t get_index(const entry_t& i) { return i; }
index_t get_coefficient(const entry_t& i) { return 1; }
entry_t make_entry(index_t _index, coefficient_t _value) { return entry_t(_index); }
void set_coefficient(entry_t& e, const coefficient_t c) {}


const entry_t& get_entry(const entry_t& e) { return e; }

typedef std::pair<value_t, index_t> diameter_index_t;
value_t get_diameter(const diameter_index_t& i) { return i.first; }
index_t get_index(const diameter_index_t& i) { return i.second; }

typedef std::pair<index_t, value_t> index_diameter_t;
index_t get_index(const index_diameter_t& i) { return i.first; }
value_t get_diameter(const index_diameter_t& i) { return i.second; }

// The basic data type for entries in a (diameter_entry_t) boundary or coefficient matrix is a tuple consisting of a simplex index (index_t), a floating point value
// (value_t) caching the diameter of the simplex with that index, and a coefficient (coeff_t) if coefficients are enabled.
struct diameter_entry_t : std::pair<value_t, entry_t> {
	using std::pair<value_t, entry_t>::pair;
	diameter_entry_t(value_t _diameter, index_t _index, coefficient_t _coefficient)
	    : diameter_entry_t(_diameter, make_entry(_index, _coefficient)) {}
	diameter_entry_t(const diameter_index_t& _diameter_index, coefficient_t _coefficient)
	    : diameter_entry_t(get_diameter(_diameter_index),
	                       make_entry(get_index(_diameter_index), _coefficient)) {}
	diameter_entry_t(const diameter_index_t& _diameter_index)
	    : diameter_entry_t(get_diameter(_diameter_index),
	                       make_entry(get_index(_diameter_index), 0)) {}
	diameter_entry_t(const index_t& _index) : diameter_entry_t(0, _index, 0) {}
};

const entry_t& get_entry(const diameter_entry_t& p) { return p.second; }
entry_t& get_entry(diameter_entry_t& p) { return p.second; }
index_t get_index(const diameter_entry_t& p) { return get_index(get_entry(p)); }
coefficient_t get_coefficient(const diameter_entry_t& p) {
	return get_coefficient(get_entry(p));
}
const value_t& get_diameter(const diameter_entry_t& p) { return p.first; }
void set_coefficient(diameter_entry_t& p, const coefficient_t c) {
	set_coefficient(get_entry(p), c);
}

template <typename Entry> struct greater_diameter_or_smaller_index_comp {
	bool operator()(const Entry& a, const Entry& b) {
		return greater_diameter_or_smaller_index(a, b);
	}
};

template <typename Entry> bool greater_diameter_or_smaller_index(const Entry& a, const Entry& b) {
	return (get_diameter(a) > get_diameter(b)) ||
	       ((get_diameter(a) == get_diameter(b)) && (get_index(a) < get_index(b)));
}

enum compressed_matrix_layout { LOWER_TRIANGULAR, UPPER_TRIANGULAR };

template < compressed_matrix_layout Layout > 
struct compressed_distance_matrix {
	std::vector<value_t> distances;
	std::vector<value_t*> rows;

	compressed_distance_matrix(std::vector< value_t >&& _distances)
	    : distances(std::move(_distances)), rows((1 + std::sqrt(1 + 8 * distances.size())) / 2) {
		assert(distances.size() == size() * (size() - 1) / 2);
		init_rows();
	}

	template <typename DistanceMatrix>
	compressed_distance_matrix(const DistanceMatrix& mat)
	    : distances(mat.size() * (mat.size() - 1) / 2), rows(mat.size()) {
		init_rows();

		for (size_t i = 1; i < size(); ++i)
			for (size_t j = 0; j < i; ++j) rows[i][j] = mat(i, j);
	}

	value_t operator()(const index_t i, const index_t j) const;
	size_t size() const { return rows.size(); }
	void init_rows();
};

typedef compressed_distance_matrix<LOWER_TRIANGULAR> compressed_lower_distance_matrix;
typedef compressed_distance_matrix<UPPER_TRIANGULAR> compressed_upper_distance_matrix;

template <> void compressed_lower_distance_matrix::init_rows() {
	value_t* pointer = &distances[0];
	for (size_t i = 1; i < size(); ++i) {
		rows[i] = pointer;
		pointer += i;
	}
}

template <> void compressed_upper_distance_matrix::init_rows() {
	value_t* pointer = &distances[0] - 1;
	for (size_t i = 0; i < size() - 1; ++i) {
		rows[i] = pointer;
		pointer += size() - i - 2;
	}
}

template <>
value_t compressed_lower_distance_matrix::operator()(const index_t i, const index_t j) const {
	return i == j ? 0 : i < j ? rows[j][i] : rows[i][j];
}

template <>
value_t compressed_upper_distance_matrix::operator()(const index_t i, const index_t j) const {
	return i == j ? 0 : i > j ? rows[j][i] : rows[i][j];
}

struct sparse_distance_matrix {
	std::vector<std::vector<index_diameter_t>> neighbors;

	index_t num_edges;

	sparse_distance_matrix(std::vector<std::vector<index_diameter_t>>&& _neighbors,
	                       index_t _num_edges)
	    : neighbors(std::move(_neighbors)), num_edges(_num_edges) {}

	template <typename DistanceMatrix>
	sparse_distance_matrix(const DistanceMatrix& mat, const value_t threshold)
	    : neighbors(mat.size()), num_edges(0) {

		for (size_t i = 0; i < size(); ++i)
			for (size_t j = 0; j < size(); ++j)
				if (i != j) {
					auto d = mat(i, j);
					if (d <= threshold) {
						++num_edges;
						neighbors[i].push_back({j, d});
					}
				}
	}

	value_t operator()(const index_t i, const index_t j) const {
		auto neighbor =
		    std::lower_bound(neighbors[i].begin(), neighbors[i].end(), index_diameter_t{j, 0});
		return (neighbor != neighbors[i].end() && get_index(*neighbor) == j)
		           ? get_diameter(*neighbor)
		           : std::numeric_limits<value_t>::infinity();
	}

	size_t size() const { return neighbors.size(); }
};

struct euclidean_distance_matrix {
	std::vector<std::vector<value_t>> points;

	euclidean_distance_matrix(std::vector<std::vector<value_t>>&& _points)
	    : points(std::move(_points)) {
		for (auto p : points) { assert(p.size() == points.front().size()); }
	}

	value_t operator()(const index_t i, const index_t j) const {
		assert(i < points.size());
		assert(j < points.size());
		return std::sqrt(std::inner_product(
		    points[i].begin(), points[i].end(), points[j].begin(), value_t(), std::plus<value_t>(),
		    [](value_t u, value_t v) { return (u - v) * (u - v); }));
	}

	size_t size() const { return points.size(); }
};

class union_find {
	std::vector<index_t> parent;
	std::vector<uint8_t> rank;

public:
	union_find(const index_t n) : parent(n), rank(n, 0) {
		for (index_t i = 0; i < n; ++i) parent[i] = i;
	}

	index_t find(index_t x) {
		index_t y = x, z;
		while ((z = parent[y]) != y) y = z;
		while ((z = parent[x]) != y) {
			parent[x] = y;
			x = z;
		}
		return z;
	}

	void link(index_t x, index_t y) {
		if ((x = find(x)) == (y = find(y))) return;
		if (rank[x] > rank[y])
			parent[y] = x;
		else {
			parent[x] = y;
			if (rank[x] == rank[y]) ++rank[y];
		}
	}
};

template <typename T> T begin(std::pair<T, T>& p) { return p.first; }
template <typename T> T end(std::pair<T, T>& p) { return p.second; }

template <typename ValueType> 
class compressed_sparse_matrix {
	std::vector<size_t> bounds;
	std::vector<ValueType> entries;

	typedef typename std::vector<ValueType>::iterator iterator;
	typedef std::pair<iterator, iterator> iterator_pair;

public:
	size_t size() const { return bounds.size(); }

	iterator_pair subrange(const index_t index) {
		return {entries.begin() + (index == 0 ? 0 : bounds[index - 1]),
		        entries.begin() + bounds[index]};
	}

	void append_column() { bounds.push_back(entries.size()); }

	void push_back(const ValueType e) {
		assert(0 < size());
		entries.push_back(e);
		++bounds.back();
	}
};

template <class Predicate>
index_t get_max(index_t top, const index_t bottom, const Predicate pred) {
	if (!pred(top)) {
		index_t count = top - bottom;
		while (count > 0) {
			index_t step = count >> 1, mid = top - step;
			if (!pred(mid)) {
				top = mid - 1;
				count -= step + 1;
			} else
				count = step;
		}
	}
	return top;
}
