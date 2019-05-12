load("//third_party/tensorflow:tensorflow.bzl",
     "tf_kernel_binary")
tf_kernel_binary(
  name = "my_matmul_op.so",
  gpu_srcs = ["my_matmul_op.cu.cc",
              "my_matmul_op.h"],
  deps = ["//twodlearn/core/cuda:matmul-pattern"],
  visibility=["//twodlearn:__subpackages__"],
  )
