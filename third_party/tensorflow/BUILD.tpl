licenses(["notice"])

cc_library(
    name = "tensorflow_headers",
    hdrs = [":tensorflow_include"],
    includes = ["tensorflow_include"],
    linkopts = ["-L%{TENSORFLOW_IMPORT_LIB_GENRULE}",
                "-ltensorflow_framework"],
    visibility = ["//visibility:public"]
)

%{TENSORFLOW_INCLUDE_GENRULE}

#
# linkopts = ["-L/home/marinodl/envs/tensorflow/lib/python3.5/site-packages/tensorflow/",
#                "-ltensorflow_framework"],
