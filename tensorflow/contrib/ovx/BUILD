# Description:
# 	 VSI OVX controller logic

package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cc_test",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "ovx_controller",
    includes = [
        "ovx_controller/src_log/include/",
        "ovx_controller/src_soc_interface/include/",
        "ovx_controller/src_impl/include/",
    ],
    srcs = [
        "ovx_controller/src_impl/ovx_controller.c",
        "ovx_controller/src_impl/ovx_controller_attr_reader.c",
        "ovx_controller/src_soc_interface/soc_interface.c",
    ],
    deps = [
        "//contrib/ovxlib:ovxlib"
    ],
)

