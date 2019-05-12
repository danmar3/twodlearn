def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        #"@local_config_cuda//cuda:using_nvcc": if_true,
        "//tools/clang_compiler:using_cuda_clang": if_true,
        "//conditions:default": if_false
    })


def cuda_default_copts():
    """Default options for all CUDA compilations."""
    return if_cuda(["-x cuda", "--cuda-gpu-arch=sm_60"])


def cuda_binary(name, srcs=None, deps=None, **kwargs):
  if not srcs:
    srcs = []
  if not deps:
    deps = []
  # Creating a native genrule.
  native.cc_binary(
      name = name,
      srcs = if_cuda(srcs),
      deps = if_cuda(deps),
      copts = cuda_default_copts(),
      linkopts=if_cuda(["-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcuda -lcudart"]),
      **kwargs
  )


def cuda_library(deps=None, cuda_deps=None, copts=None, **kwargs):
  """Generate a cc_library with a conditional set of CUDA dependencies.
  """
  if not deps:
    deps = []
  if not cuda_deps:
    cuda_deps = []
  if not copts:
    copts = []

  native.cc_library(
      deps=deps + if_cuda(cuda_deps),
      copts=copts + cuda_default_copts(),
      **kwargs)
