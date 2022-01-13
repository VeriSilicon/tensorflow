licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

cc_library(
    name = "vsi_platform_id",
    srcs = ["vsi_platform_id.cc"],
    hdrs = ["vsi_platform_id.h"],
    deps = ["//tensorflow/core:stream_executor_headers_lib"] + if_static(
        ["@com_google_protobuf//:protobuf"],
        ["@com_google_protobuf//:protobuf_headers"],
    ),
)

cc_library(
    name = "vsi_pass",
    srcs = ["passes/InsertTranspose.cc"],
    hdrs = ["passes/InsertTranspose.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        ],
)

cc_library(
    name = "vsi_compiler",
    srcs = ["vsi_compiler.cc"],
    hdrs = ["vsi_compiler.h"],
    deps = [
        ":vsi_executable",
        "vsi_pass",
        ":vsi_platform_id",
        "vsi_platform",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:cholesky_expander",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//tensorflow/compiler/xla/service:map_inliner",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:triangular_solve_expander",
        "//tensorflow/compiler/xla/service:while_loop_simplifier",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "vsi_executor",
    srcs = ["vsi_executor.cc"],
    hdrs = ["vsi_executor.h",
            "vsi_utils.h"
    ],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/stream_executor/host:host_stream",
        "//tensorflow/stream_executor/host:host_timer",
        "@TIM_VX//:tim-vx_interface",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vsi_executable",
    srcs = ["vsi_executable.cc"],
    hdrs = ["vsi_executable.h"],
    deps = [
        ":vsi_executor",
        ":visitor_base",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@TIM_VX//:libtim-vx.so"
    ],
)

cc_library(
    name = "vsi_platform",
    srcs = ["vsi_platform.cc"],
    hdrs = ["vsi_platform.h",
            "vsi_utils.h"
    ],
    deps = [
        ":vsi_executor",
        ":vsi_platform_id",
        "//tensorflow/core:stream_executor_headers_lib",
        "@TIM_VX//:libtim-vx.so"
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "vsi_transfer_manager",
    srcs = ["vsi_transfer_manager.cc"],
    hdrs = ["vsi_transfer_manager.h"],
    deps = [
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        ":vsi_platform_id",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "vsi_xla_device",
    srcs = [
        "xla_vsi_npu_device.cc",
    ],
    hdrs = [],
    data = [
    ],
    linkstatic = 1,
    deps = [
        ":vsi_platform",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:no_op",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "visitor_base",
    srcs = [
        "visitors/visitor_base.cc",
    ],
    hdrs = ["visitors/visitor_base.h"],
    deps = [
        ":vsi_executor",
        "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "//tensorflow/compiler/xla/service:dynamic_dimension_inference",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_value",
        "//tensorflow/compiler/xla/service:hlo_buffer",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@TIM_VX//:libtim-vx.so"
    ],
)
cc_library(
    name = "vsi_plugin",
    deps = [
        ":vsi_executable",
        ":vsi_platform",
        "vsi_compiler",
        "vsi_transfer_manager",
        ":visitor_base",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "vsi_lib",
    srcs =[
        "vsi_compiler.cc",
        "vsi_executable.cc",
        "vsi_executor.cc",
        "vsi_platform.cc",
        "vsi_platform_id.cc",
        "vsi_transfer_manager.cc",
        "xla_vsi_npu_device.cc",
        "visitors/visitor_base.cc",
    ],
    hdrs = [
        "vsi_compiler.h",
        "vsi_executable.h",
        "vsi_executor.h",
        "vsi_platform.h",
        "vsi_platform_id.h",
        "vsi_transfer_manager.h",
        "vsi_utils.h",
        "visitors/visitor_base.h"
    ],
    deps = [
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "//tensorflow/compiler/xla/service:dynamic_dimension_inference",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_value",
        "//tensorflow/compiler/xla/service:hlo_buffer",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
        "//tensorflow/core:lib",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        #"//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:no_op",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@TIM_VX//:libtim-vx.so"        
    ]
)

