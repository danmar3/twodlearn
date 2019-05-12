cc_library(
    name = "cuda-error",
    hdrs = ["cuda_error.cu.h"],
    deps = ["@cuda_linux//:cuda-lib"],
    visibility = ["//twodlearn:__pkg__"]
)

cc_library(
    name = "twin-object",
    hdrs = ["twin_object.cu.h"],
    deps = [":cuda-error"],
    visibility = ["//twodlearn:__pkg__"]
)

cc_library(
    name = "eigen-cuda",
    hdrs = ["eigen_cuda.cu.h"],
    deps = [":twin-object",
            "@eigen_src//:eigen3"],
    visibility = ["//twodlearn/core:__subpackages__"]
)

cc_library(
    name = "elementwise-pattern",
    hdrs = ["elementwise_pattern.cu.h"],
    deps = ["@cuda_linux//:cuda-lib"],
    visibility = ["//twodlearn/core:__subpackages__"]
)

cc_library(
    name = "matmul-pattern",
    hdrs = ["matmul_pattern.cu.h"],
    deps = ["@cuda_linux//:cuda-lib"],
    visibility = ["//twodlearn/core:__subpackages__"]
)
