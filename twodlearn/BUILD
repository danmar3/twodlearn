py_library(
  name="core",
  #srcs=["__init__.py"],
  srcs=glob(["*.py", "core/*.py"]),
  srcs_version = "PY2AND3",
  deps=["//twodlearn/core/ops:kernels"],
  visibility=["//visibility:public"],
  )

py_library(
  name="bayes",
  srcs=glob(["bayesnet/*.py"]),
  srcs_version = "PY2AND3",
  visibility=["//visibility:public"],
  )

py_library(
  name="datasets",
  srcs=glob(["datasets/*.py"]),
  srcs_version = "PY2AND3",
  visibility=["//visibility:public"],
  )

py_library(
  name="templates",
  srcs=glob(["templates/*.py"]),
  srcs_version = "PY2AND3",
  visibility=["//visibility:public"],
  )

py_library(
  name="tests",
  deps=["//twodlearn/core/ops:kernel_tests"],
  visibility=["//visibility:public"],
  )
