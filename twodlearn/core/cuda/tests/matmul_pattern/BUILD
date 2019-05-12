load("//tools/clang_compiler:cuda.bzl", "cuda_binary")
cuda_binary(
  name = "matmul-pattern-test",
  srcs = ["matmul_pattern.cu.cc"],
  deps = ["//twodlearn/core/cuda:eigen-cuda",
          "//twodlearn/core/cuda:matmul-pattern"]
  )
