import importlib.metadata
__version__ = importlib.metadata.version("spirit")


## Based on Numpy's usage: https://github.com/numpy/numpy/blob/v1.25.0/numpy/lib/utils.py#L75-L101
def get_include():
  """Return the directory that contains the packages \\*.h header files.

  Extension modules that need to compile against the packages exported header files should use this
  function to locate the appropriate include directory.

  Notes: 
    When using `distutils`, for example in `setup.py`:
      ```python
      import spirit
      ...
      Extension('extension_name', ..., include_dirs=[spirit.get_include()])
      ...
      ```
    Or with `meson-python`, for example in `meson.build`:
      ```meson
      ...
      run_command(py, ['-c', 'import spirit; print(spirit.get_include())', check : true).stdout().strip()
      ...
      ```
  """
  import os 
  d = os.path.join(os.path.dirname(__file__), 'include')
  return d

