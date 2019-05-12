# clang++-6.0 -std=c++11 -x cuda --cuda-gpu-arch=sm_60 -c twinmat_test.cc -isystem /usr/local/cuda/include/
load("//tools/clang_compiler:cuda.bzl", "cuda_binary")
cuda_binary(
  name="elementwise-pattern-test",
  srcs = ["elementwise_pattern.cu.cc"],
  deps = ["//twodlearn/core/cuda:eigen-cuda",
          "//twodlearn/core/cuda:elementwise-pattern"]
  )
