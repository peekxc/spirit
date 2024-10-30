import numpy as np

def min_rolling_distance(x: np.ndarray, y: np.ndarray, method: str = "fft"):
  assert len(x) == len(y), "Must be same same length"

  ## Circular cross-correlation via FFT
  x_norm_sq, y_norm_sq = np.sum(x**2), np.sum(y**2)  
  cross_corr = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y[::-1], len(x))).real
  
  ## Compute L2 norm squared distances
  norms_squared = x_norm_sq + y_norm_sq - 2 * cross_corr
  return np.sqrt(np.min(norms_squared))

## This on doesn't work
## Though see: https://www.mathworks.com/help/signal/ref/xcorr2.html
def min_rolling_distance_matrix(X, Y):
  ## Perform 2D circular cross-correlation using FFT
  X_norm_sq, Y_norm_sq, = np.sum(X**2), np.sum(Y**2)
  cross_corr = np.fft.ifft2(np.fft.fft2(X) * np.fft.fft2(Y[::-1], s=X.shape)).real
  
  ## Compute L2 norm squared distances for each shift
  norms_squared = X_norm_sq + Y_norm_sq - 2 * cross_corr
  return np.sqrt(np.min(norms_squared))

# X = np.random.uniform(size=(100,5))
# Y = np.random.uniform(size=(100,5))
# np.linalg.norm(X - Y, "fro")
# np.min([np.linalg.norm(np.roll(X, i, axis=0) - Y, "fro") for i in range(len(X))])
# min_rolling_distance_matrix(X, Y)