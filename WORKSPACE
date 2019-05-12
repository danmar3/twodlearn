workspace(name = 'twodlearn_root')

new_local_repository(
    name = "cuda_linux",
    path = "/usr/local/cuda",
    build_file_content = """
cc_library(
    name = "cuda-lib",
    srcs = ["lib64/libcublas.so",
            "lib64/libcudart.so",
            "lib64/libcufft.so",
            "lib64/libcurand.so"],
    hdrs = glob(["include/**/*.h",
                 "include/**/*.hpp",
                 "include/**/*.inl"]),
    includes = ["include/"],
    linkopts = ["-lcudart",
                "-Wl,-rpath,/usr/local/cuda/lib64"],
    visibility = ["//visibility:public"]
)
    """
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
    name = "eigen_src",
    #build_file = "third_party/eigen/eigen.BUILD",
    remote = "https://github.com/eigenteam/eigen-git-mirror.git",
    tag = "3.2.10",
    build_file_content = """
cc_library(
    name = "eigen3",
    hdrs = glob(["Eigen/**"]),
    includes = ["./"],
    visibility = ["//visibility:public"]
)""",
)


new_local_repository(
    name = "local_tensorflow",
    #path = "/home/marinodl/envs/jmodelica2/local/lib/python2.7/site-packages/tensorflow/",
    path = "/home/marinodl/envs/tensorflow/lib/python3.5/site-packages/tensorflow/",
    build_file_content = """
cc_library(
    name = "tensorflow_headers",
    hdrs = glob(["include/**"]),
    includes = ["include/"],
    linkopts = ["-L/home/marinodl/envs/tensorflow/lib/python3.5/site-packages/tensorflow/",
                "-ltensorflow_framework"],
    visibility = ["//visibility:public"]
)
    """
)

load("//third_party/tensorflow:tensorflow.bzl", "local_tensorflow_rep")
local_tensorflow_rep(
  name="auto_tensorflow")
# # ------------------ Tensorflow ------------- #
# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)
#
# #  load tensorflow repository
# load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
# git_repository(
#     name = "org_tensorflow",
#     remote = "https://github.com/tensorflow/tensorflow.git",
#     tag = "v1.9.0",
# )
#
# load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')
# tf_workspace(path_prefix="", tf_repo_name = "org_tensorflow")
