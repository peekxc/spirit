#include <vector> 
#include <cinttypes>
 
using index_t = int64_t;
using std::vector;
using vector2D = std::vector< std::vector< index_t > >;

// Baseline from: https://stackoverflow.com/questions/44718971/calculate-binomial-coffeficient-very-reliably
// Requires O(min{k,n-k}), uses pascals triangle approach (+ degenerate cases)
__device__ inline size_t binom(size_t n, size_t k) noexcept {
  return
    (k > n) ? 0 :                  // out of range
    (k == 0 || k == n) ? 1 :       // edge
    (k == 1 || k == n-1) ? n :     // first
    (k+k < n) ?                    // recursive:
    (binom(n-1,k-1) * n)/k :       //  path to k=1   is faster
    (binom(n-1,k) * n)/(n-k);      //  path to k=n-1 is faster
}

// Table to cache low values of the binomial coefficient
// template< typename value_t = index_t >
// struct BinomialCoefficientTable {
//   size_t pre_n = 0;
//   size_t pre_k = 0; 
//   vector< vector< value_t > > BT;

//   // Safe, general binomial coefficient; uses cached table if possible 
//   value_t operator()(const index_t n, const index_t k) const {
//     if (n <= pre_n && k <= pre_k){ return BT[k][n]; } 				// runtime computed extension table
//     return static_cast< value_t >(binom(n,std::min(k,n-k)));
//   }
//   // Fast, unsafe access to a precompute table
//   constexpr auto at(index_t n, index_t k) const noexcept -> index_t {
//     return BT[k][n];
//   }

//   // Precompute a *larger* table of binomial coefficients
  void precompute(index_t n, index_t k){
    pre_n = n;
    pre_k = k;
    BT = std::vector< std::vector< value_t > >(k + 1, std::vector< value_t >(n + 1, 0));
    for (index_t i = 0; i <= n; ++i) {
      BT[0][i] = 1;
      for (index_t j = 1; j < std::min(i, k + 1); ++j){
        BT[j][i] = binom(i,j); // BT[j - 1][i - 1] + BT[j][i - 1];
      }
      if (i <= k) { BT[i][i] = 1; };
    }
  }
// }; // BinomialCoefficientTable

__device__ index_t get_max(index_t top, index_t bottom, const index_t r, const index_t m, const vector2D& BT) noexcept {
  if (!(BT[m][bottom] <= r)) { return bottom; }
  index_t size = (top - bottom);
  while (size > 0){
    index_t step = size >> 1;
    index_t mid = top - step;
    if (BT[m][mid] <= r){
    // if (!pred(mid)){
      top = mid - 1;
      size -= step + 1;
    } else {
      size = step;
    }
  }
  return top;
}

__device__ index_t get_max_vertex(const index_t r, const index_t m, const index_t n, const vector2D& BT) noexcept {
  index_t k_lb = m - 1;
  return 1 + get_max(n, k_lb, r, m, BT);
}

// Enumerates the facets on the boundary of 'simplex'
template< typename Lambda > 
__device__ void enum_boundary(const int n, const int simplex, const int dim, const vector2D& BT, Lambda&& f) {
  index_t idx_below = simplex;
  index_t idx_above = 0; 
  index_t j = n - 1;
  bool cont_enum = true; 
  for (index_t k = dim; k >= 0 && cont_enum; --k){
    j = get_max_vertex(idx_below, k + 1, j, BT) - 1; // NOTE: Danger!
    index_t c = BT[k+1][j];
    index_t face_index = idx_above - c + idx_below;
    idx_below -= c;
    idx_above += BT[k][j];
    cont_enum = f(face_index);
  }
}

__device__ void k_boundary(const index_t n, const index_t simplex, const index_t dim, const vector2D& BT, index_t* br) {
  index_t idx_below = simplex;
  index_t idx_above = 0; 
  index_t j = n - 1;
  bool cont_enum = true; 
  for (index_t k = dim; k >= 0 && cont_enum; --k){
    j = get_max_vertex(idx_below, k + 1, j, BT) - 1; // NOTE: Danger!
    index_t c = BT[k+1][j];
    index_t face_index = idx_above - c + idx_below;
    idx_below -= c;
    idx_above += BT[k][j];
    br[dim-k] = face_index;
  }
}

void compute_deg_full(const int n, const int k, const vector2D& BT, std::vector< size_t >& deg){
  for (size_t r = 0; r < BT[k][n]; ++r){
    // enum_boundary(n, r, k - 1, BT, [&](int pr){ deg[pr] += 1; return true; });
    index_t ps[3]; 
    k_boundary(n, r, k - 1, BT, ps);
    for(size_t i = 0; i < 3; ++i){
      deg[ps[i]] += 1;
    }
  }
}

// Kernel for adding the boundary/coboundary of r 
// k should == array size 
__global__ void add_boundaries(const index_t k, const index_t n, const index_t N, const index_t* sgn_pattern, const vector2D& BT, const float* x, float *y)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N){
    index_t ps[3]; 
    k_boundary(n, tid, k - 1, BT, &ps);
    int cc = 0; 
    for (int ii = 0; ii < k; ++ii){
      for (int jj = ii + 1; jj < k; ++jj){
        atomicAdd(y + ps[ii], sgn_pattern[cc] * x[ps[jj]]);
        ++cc;
      }
    }
  }  
}

void compute_deg_full(const int n, const int k, const index_t* BT, index_t* deg){
  for (size_t r = 0; r < binom(n, k); ++r){
    index_t ps[3];
    k_boundary(n, r, k - 1, BT, ps);
    for(size_t i = 0; i < 3; ++i){
      deg[ps[i]] += 1;
    }
  }
}