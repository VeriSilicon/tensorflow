# Prepare

## python

```bash
sudo apt install python3-dev python3-pip
pip install -U --user pip numpy wheel
pip install -U --user keras_preprocessing --no-deps
```

## bazelisk

https://github.com/bazelbuild/bazelisk

```bash
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/<latest release>/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```

# Build

get source code:

```bash
git clone https://github.com/Verisilicon/tensorflow.git
git checkout xla_npu_v270
```

build:

```bash
./configure
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

If your PC don't have enough memory(32GB), you can limit your threads like this:

```bash
bazel build --jobs 2 --config=opt //tensorflow/tools/pip_package:build_pip_package
```

If you meet error, you can get the verbose like this:

```bash
bazel build --verbose_failures --config=opt //tensorflow/tools/pip_package:build_pip_package
```

unit test build:

```bash
bazel build tensorflow/compiler/xla/tests:convolution_variants_test
./bazel-bin/tensorflow/compiler/xla/tests/convolution_variants_test_cpu --gtest_filter=*BackwardInputLowPaddingLessThanHighPadding*
```

# Install

```bash
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg # build wheel
pip install /tmp/tensorflow_pkg/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl
```

# Test

model test:

```bash
# TF envs
export TF_CPP_MAX_VLOG_LEVEL=1 # the higher level, the more logs.
export TF_CPP_MIN_LOG_LEVEL=0 # the lower level, the more logs.
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" # enable XLA
export CUDA_VISIBLE_DEVICES=

# VSI envs
export VIVANTE_SDK_DIR=<tf path>/bazel-tensorflow/external/TIM_VX/prebuilt-sdk/x86_64_linux
export LD_LIBRARY_PATH=<tf path>/bazel-tensorflow/external/TIM_VX/prebuilt-sdk/x86_64_linux/lib:$VIVANTE_SDK_DIR/drivers:$LD_LIBRARY_PATH
export VSI_NN_LOG_LEVEL=5
export VIV_VX_DEBUG_LEVEL=1
export VSIMULATOR_CONFIG=VIP8000NANOSI_PLUS_PID0X9F

cd python_test
python3 lenet_xla.py
```
