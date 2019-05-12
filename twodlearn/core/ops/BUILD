py_library(
  name="kernels",
  srcs=["__init__.py"],
  data = ["//twodlearn/core/ops/kernels:my_matmul_op.so"],
  visibility=["//twodlearn:__subpackages__"],
  )


py_library(
  name="kernel_tests",
  srcs=["tests/my_matmul_test2.py"],
  deps=[":kernels"],
  visibility=["//twodlearn:__subpackages__"],
  )

py_test(
  name="my_matmul_test2",
  srcs=["tests/my_matmul_test2.py"],
  deps=[":kernels"],
  visibility=["//twodlearn:__subpackages__"],
  )
