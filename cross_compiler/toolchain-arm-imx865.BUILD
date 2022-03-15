package(default_visibility = ['//visibility:public'])

filegroup(
  name = 'gcc',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-gcc',
  ],
)

filegroup(
  name = 'ar',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-ar',
  ],
)

filegroup(
  name = 'ld',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-ld',
  ],
)

filegroup(
  name = 'nm',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-nm',
  ],
)

filegroup(
  name = 'objcopy',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-objcopy',
  ],
)

filegroup(
  name = 'objdump',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-objdump',
  ],
)

filegroup(
  name = 'strip',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-strip',
  ],
)

filegroup(
  name = 'as',
  srcs = [
    'x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-as',
  ],
)

filegroup(
  name = 'compiler_pieces',
  srcs = glob([
    'cortexa53-crypto-poky-linux/usr/lib/**',
    "cortexa53-crypto-poky-linux/usr/lib/aarch64-poky-linux/10.2.0/**",
  ]),
)

filegroup(
  name = 'compiler_components',
  srcs = [
    ':gcc',
    ':ar',
    ':ld',
    ':nm',
    ':objcopy',
    ':objdump',
    ':strip',
    ':as',
  ],
)
