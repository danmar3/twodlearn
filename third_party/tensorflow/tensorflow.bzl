_BAZEL_SH = "BAZEL_SH"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_PYTHON_LIB_PATH = "PYTHON_LIB_PATH"


load("//tools/clang_compiler:cuda.bzl",
     "if_cuda", "cuda_binary", "cuda_default_copts")

def tf_default_copts():
  return ["-D_GLIBCXX_USE_CXX11_ABI=0"]

def tf_cuda_default_copts():
  return cuda_default_copts() + tf_default_copts() + ["-DGOOGLE_CUDA=1"]

def tf_cuda_library(deps=None, cuda_deps=None, copts=None, **kwargs):
  """Generate a cc_library with a conditional set of CUDA dependencies.
  When the library is built with --config=cuda_clang:
  - Both deps and cuda_deps are used as dependencies.
  - The cuda runtime is added as a dependency (if necessary).
  - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts.
  - In addition, when the library is also built with TensorRT enabled, it
      additionally passes -DGOOGLE_TENSORRT=1 to the list of copts.
  Args:
  - cuda_deps: BUILD dependencies which will be linked if and only if:
      '--config=cuda' is passed to the bazel command line.
  - deps: dependencies which will always be linked.
  - copts: copts always passed to the cc_library.
  - kwargs: Any other argument to cc_library.
  """
  if not deps:
    deps = []
  if not cuda_deps:
    cuda_deps = []
  if not copts:
    copts = []

  native.cc_library(
      deps=deps + if_cuda(cuda_deps),
      copts=copts + tf_cuda_default_copts(),
      **kwargs)

def tf_kernel_binary(name, srcs=[], gpu_srcs=[], deps=[], copts=[], **kwargs):
  if gpu_srcs:
    native.cc_binary(
      name=name,
      srcs=srcs + gpu_srcs,
      #deps=deps + ["@local_tensorflow//:tensorflow_headers"],
      deps=deps + ["@auto_tensorflow//:tensorflow_headers"],
      copts=copts + tf_cuda_default_copts(),
      linkopts=if_cuda(["-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcuda -lcudart"]),
      linkshared=1,
      **kwargs
      )
  else:
    native.cc_binary(
      name=name,
      srcs=srcs,
      deps=deps,
      copts=copts + tf_default_copts(),
      linkshared=1,
      **kwargs
      )


load("//third_party/tensorflow/py:python_configure.bzl",
     "get_python_bin", "get_bash_bin", "symlink_genrule_for_dir")

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl
  repository_ctx.template(
      out,
      Label("//third_party/tensorflow:%s.tpl" % tpl),
      substitutions)

def _fail(msg):
  """Output failure message when auto configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("%sTensorflow Configuration Error:%s %s\n" % (red, no_color, msg))


def _get_tensorflow_cflags(repository_ctx, python_bin):
  """Gets the tensorflow include path."""
  print_lib = ("<<END\n" +
      "from __future__ import print_function\n" +
      "try:\n" +
      "  import tensorflow as tf\n" +
      "  print(' '.join(tf.sysconfig.get_compile_flags()))\n" +
      "except NameError:\n" +
      "  pass\n" +
      "END")
  cmd = '%s - %s' % (python_bin, print_lib)
  result = repository_ctx.execute([get_bash_bin(repository_ctx), "-c", cmd])
  return result.stdout.strip('\n')


def _get_tensorflow_include(repository_ctx, python_bin):
  """Gets the tensorflow include path."""
  print_lib = ("<<END\n" +
      "from __future__ import print_function\n" +
      "try:\n" +
      "  import tensorflow as tf\n" +
      "  print(tf.sysconfig.get_include())\n" +
      "except NameError:\n" +
      "  pass\n" +
      "END")
  cmd = '%s - %s' % (python_bin, print_lib)
  result = repository_ctx.execute([get_bash_bin(repository_ctx), "-c", cmd])
  return result.stdout.strip('\n')

def _get_tensorflow_lib(repository_ctx, python_bin):
  """Gets the tensorflow include path."""
  print_lib = ("<<END\n" +
      "from __future__ import print_function\n" +
      "try:\n" +
      "  import tensorflow as tf\n" +
      "  print(tf.sysconfig.get_lib())\n" +
      "except NameError:\n" +
      "  pass\n" +
      "END")
  cmd = '%s - %s' % (python_bin, print_lib)
  result = repository_ctx.execute([get_bash_bin(repository_ctx), "-c", cmd])
  return result.stdout.strip('\n')


def _get_tensorflow_linkflags(repository_ctx, python_bin):
  """Gets the tensorflow lib path."""
  print_lib = ("<<END\n" +
      "from __future__ import print_function\n" +
      "import tensorflow as tf\n" +
      "print(' '.join(tf.sysconfig.get_link_flags()))\n" +
      "END")
  cmd = '%s - %s' % (python_bin, print_lib)
  result = repository_ctx.execute([get_bash_bin(repository_ctx), "-c", cmd])
  return result.stdout.strip('\n')


def _local_tensorflow_rep(repository_ctx):
  #repository_ctx.execute(
  #  "echo $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')")
  python_bin = get_python_bin(repository_ctx)
  tensorflow_include = _get_tensorflow_include(repository_ctx, python_bin)
  tensorflow_lib = _get_tensorflow_lib(repository_ctx, python_bin)
  tf_include_rule = symlink_genrule_for_dir(
    repository_ctx, tensorflow_include, 'tensorflow_include', 'tensorflow_include'
    )
  print(tensorflow_include)
  # _fail("not running test")
  _tpl(repository_ctx, "BUILD", {
      "%{TENSORFLOW_INCLUDE_GENRULE}": tf_include_rule,
      "%{TENSORFLOW_IMPORT_LIB_GENRULE}": tensorflow_lib,
  })


local_tensorflow_rep = repository_rule(
    implementation = _local_tensorflow_rep,
)
