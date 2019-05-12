package(default_visibility = ['//visibility:public'])

config_setting(
    name = "using_clang",
    values = {
        "define": "using_clang=true",
    },
)

config_setting(
    name = "using_cuda_clang",
    values = {
        "define": "using_cuda_clang=true",
    },
)

# Equivalent to using_clang && -c opt.
config_setting(
    name = "using_clang_opt",
    values = {
        "define": "using_clang=true",
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "using_cuda_clang_opt",
    values = {
        "define": "using_cuda_clang=true",
        "compilation_mode": "opt",
    },
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
         "k8|clang": ":linux_x64_clang_toolchain",
    },
)
filegroup(name = "empty")
cc_toolchain(
    name = "linux_x64_clang_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "asmjs",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 0,
)
