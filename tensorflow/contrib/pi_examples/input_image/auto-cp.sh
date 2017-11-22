#!/usr/bin/env bash
root_dir=/home/kerner/work/github/BasePlate/base/platform
# Linux X64
# recognize module library
cp tensorflow/contrib/pi_examples/input_image/gen/lib/libpi_module_recognize.a ${root_dir}/Linux-X64/lib/
# tensorflow core library
cp tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a  ${root_dir}/Linux-X64/lib/
# nsync library
cp tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/nsync.a  ${root_dir}/Linux-X64/lib/libnsync.a
# protobuf library
cp tensorflow/contrib/makefile/gen/protobuf-host/lib/libprotobuf.a  ${root_dir}/Linux-X64/lib/
# include 
tmp=${root_dir}/Linux-X64/include/recognize
if [ ! -d "$tmp" ]; then
mkdir "$tmp"
fi
cp tensorflow/contrib/pi_examples/input_image/image_recognize.h ${root_dir}/Linux-X64/include/recognize/

# Android X64
# recognize module library
cp tensorflow/contrib/pi_examples/input_image/gen/lib/android_arm64-v8a/libpi_module_recognize.a ${root_dir}/Android-ARM64/lib/
# tensoflow core library
cp tensorflow/contrib/makefile/gen/lib/android_arm64-v8a/libtensorflow-core.a  ${root_dir}/Android-ARM64/lib/
# nsync library
cp tensorflow/contrib/makefile/downloads/nsync/builds/arm64-v8a.android.c++11/nsync.a  ${root_dir}/Android-ARM64/lib/libnsync.a
# protobuf library
cp tensorflow/contrib/makefile/gen/protobuf/lib/libprotobuf.a  ${root_dir}/Android-ARM64/lib/
# include 
tmp=${root_dir}/Android-ARM64/include/recognize
if [ ! -d "$tmp" ]; then
mkdir "$tmp"
fi
cp tensorflow/contrib/pi_examples/input_image/image_recognize.h ${root_dir}/Android-ARM64/include/recognize/
