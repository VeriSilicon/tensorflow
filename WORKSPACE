workspace(name = "org_tensorflow")

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "TIM_VX",
    remote = "https://github.com/VeriSilicon/TIM-VX.git",
    branch = "main",
    patches = ["//tensorflow/compiler/plugin/vsi/driver:0001-patch-for-TF-XLA.patch"],
    patch_args = ["-p1"],
    verbose = True,
)

# local_repository(
#     name = "TIM_VX",
#     path = "tensorflow/compiler/plugin/vsi/driver/TIM-VX",
# )
