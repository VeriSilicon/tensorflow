export NCCL_SOCKET_IFNAME=enp
export NCCL_DEBUG=INFO
export TF_DUMP_GRAPH_PREFIX=./dump_graph_gpu0
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_clustering_debug"
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./dump_graph"
python test.py
# python lenet_xla.py

