# This Makefile compiles the label_image example for the Raspberry Pi.
# See tensorflow/contrib/pi_examples/README.md for full build instructions.

# Find where we're running from, so we can store generated files here.
SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# The location of the tensorflow/contrib/makefile directory.
TFMAKEFILE_DIR := $(SCRIPT_DIR)/../../makefile

# Where compiled objects are stored.
GENDIR := $(SCRIPT_DIR)/gen/
OBJDIR := $(GENDIR)obj/
LIBDIR := $(GENDIR)lib/
BINDIR := $(GENDIR)bin/

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = WINDOWS
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	        HOST_OS := LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := OSX
	endif
endif

# The expected locations of the TensorFlow library.
TFLIBDIR := $(TFMAKEFILE_DIR)/gen/lib
TFLIBS := $(TFLIBDIR)/libtensorflow-core.a

# Where the downloads have been stored.
DOWNLOADSDIR := $(TFMAKEFILE_DIR)/downloads

# The location of the compiled protobuf headers generated by TensorFlow.
PBTGENDIR := $(TFMAKEFILE_DIR)/gen/proto_text/
PROTOGENDIR := $(TFMAKEFILE_DIR)/gen/proto/
PROTOBUFGENDIR := $(TFMAKEFILE_DIR)/gen/protobuf-host
PROTOBUFLIBS := $(PROTOBUFGENDIR)/lib/libprotobuf.a

# The location of nsync
NSYNCDIR := $(TFMAKEFILE_DIR)/downloads/nsync/builds/default.linux.c++11
NSYNCLIBS := $(NSYNCDIR)/libnsync.a


# Settings for the target compiler.
CXX := gcc
OPTFLAGS := -O0
CXXFLAGS := --std=c++11 $(OPTFLAGS)
LDFLAGS := \
-L/usr/local/lib \
-L$(TFLIBDIR) \
-L$(NSYNCDIR) \
-L$(PROTOBUFGENDIR)/lib \
-Wl,--no-whole-archive
INCLUDES := \
-I/usr/local/include \
-I. \
-I$(DOWNLOADSDIR) \
-I$(DOWNLOADSDIR)/eigen/ \
-I$(TFMAKEFILE_DIR)/downloads/nsync/public \
-I$(PROTOBUFGENDIR)/include/ \
-I$(PROTOGENDIR) \
-I$(PBTGENDIR)
LIBS := \
-Wl,--allow-multiple-definition \
-Wl,--whole-archive \
-ltensorflow-core \
-Wl,--no-whole-archive \
-lstdc++ \
-lprotobuf \
-lnsync \
-ldl \
-lpthread \
-lm \
-ljpeg \
-lz
LIBFLAGS :=

AR := ar
ARFLAGS := -r

# Set up Android building
ifeq ($(TARGET),ANDROID)

OBJDIR := $(OBJDIR)android_$(ANDROID_ARCH)/
LIBDIR := $(LIBDIR)android_$(ANDROID_ARCH)/
BINDIR := $(BINDIR)android_$(ANDROID_ARCH)/
DEPDIR := $(DEPDIR)android_$(ANDROID_ARCH)/

# The expected locations of the TensorFlow library.
TFLIBDIR := $(TFMAKEFILE_DIR)/gen/lib/android_$(ANDROID_ARCH)
TFLIBS := $(TFLIBDIR)/libtensorflow-core.a

# Where the downloads have been stored.
DOWNLOADSDIR := $(TFMAKEFILE_DIR)/downloads

# The location of the compiled protobuf headers generated by TensorFlow.
PBTGENDIR := $(TFMAKEFILE_DIR)/gen/proto_text/
PROTOGENDIR := $(TFMAKEFILE_DIR)/gen/proto/
PROTOBUFGENDIR := $(TFMAKEFILE_DIR)/gen/protobuf_android/$(ANDROID_ARCH)
PROTOBUFLIBS := $(PROTOBUFGENDIR)/lib/libprotobuf.a

# The location of nsync
NSYNCDIR := $(TFMAKEFILE_DIR)/downloads/nsync/builds/$(ANDROID_ARCH).android.c++11
NSYNCLIBS := $(NSYNCDIR)/libnsync.a

OS_PATH :=
ifeq ($(HOST_OS),LINUX)
	OS_PATH=linux
endif
ifeq ($(HOST_OS),OSX)
	OS_PATH=darwin
endif
ifeq ($(HOST_OS),WINDOWS)
    $(error "windows is not supported.")
endif

ifndef NDK_ROOT
    $(error "NDK_ROOT is not defined.")
endif

# android CPU architectures
# default architectures
ANDROID_ARCH := arm64-v8a

# arm64-v8a
ifeq ($(ANDROID_ARCH),arm64-v8a)
TOOLCHAIN := aarch64-linux-android-4.9
SYSROOT_ARCH := arm64
BIN_PREFIX := aarch64-linux-android
MARCH_OPTION :=
endif

# armeabi
ifeq ($(ANDROID_ARCH),armeabi)
TOOLCHAIN := arm-linux-androideabi-4.9
SYSROOT_ARCH := arm
BIN_PREFIX := arm-linux-androideabi
MARCH_OPTION :=
endif

# armeabi-v7a
ifeq ($(ANDROID_ARCH),armeabi-v7a)
TOOLCHAIN := arm-linux-androideabi-4.9
SYSROOT_ARCH := arm
BIN_PREFIX := arm-linux-androideabi
MARCH_OPTION := -march=armv7-a -mfloat-abi=softfp -mfpu=neon
endif


CXX := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(OS_PATH)-x86_64/bin/$(BIN_PREFIX)-g++
CC := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(OS_PATH)-x86_64/bin/$(BIN_PREFIX)-gcc

CXXFLAGS +=\
--sysroot $(NDK_ROOT)/platforms/android-21/arch-$(SYSROOT_ARCH) \
-Wno-narrowing \
-fomit-frame-pointer \
$(MARCH_OPTION) \
-fPIE

INCLUDES = \
-I$(NDK_ROOT)/sources/android/support/include \
-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/include \
-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH)/include \
-I. \
-I$(SCRIPT_DIR)/jpeg/include \
-I$(DOWNLOADSDIR) \
-I$(DOWNLOADSDIR)/eigen/ \
-I$(TFMAKEFILE_DIR)/downloads/nsync/public \
-I$(PROTOBUFGENDIR)/include/ \
-I$(PROTOGENDIR) \
-I$(PBTGENDIR)

LIBS := \
-ltensorflow-core \
-lgnustl_static \
-lstdc++ \
-lprotobuf \
-lnsync \
-ljpeg \
-llog \
-lz \
-ldl \
-lm


LD := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(OS_PATH)-x86_64/$(BIN_PREFIX)/bin/ld

LDFLAGS := \
$(MARCH_OPTION) \
-L$(TFLIBDIR) \
-L$(NSYNCDIR) \
-L$(PROTOBUFGENDIR)/lib \
-L$(SCRIPT_DIR)/jpeg/lib/$(ANDROID_ARCH) \
-L$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH) \
-fPIE \
-pie \
-v

AR := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(OS_PATH)-x86_64/bin/$(BIN_PREFIX)-ar
ARFLAGS := r
LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive

endif  # ANDROID

# Set up ZULUKO building
ifeq ($(TARGET),ZULUKO)

OBJDIR := $(OBJDIR)linux_zuluko/
LIBDIR := $(LIBDIR)linux_zuluko/
BINDIR := $(BINDIR)linux_zuluko/
DEPDIR := $(DEPDIR)linux_zuluko/

BIN_PREFIX := arm-openwrt-linux

# The expected locations of the TensorFlow library.
TFLIBDIR := $(TFMAKEFILE_DIR)/gen/lib/linux_zuluko
TFLIBS := $(TFLIBDIR)/libtensorflow-core.a

# Where the downloads have been stored.
DOWNLOADSDIR := $(TFMAKEFILE_DIR)/downloads

# The location of the compiled protobuf headers generated by TensorFlow.
PBTGENDIR := $(TFMAKEFILE_DIR)/gen/proto_text/
PROTOGENDIR := $(TFMAKEFILE_DIR)/gen/proto/
PROTOBUFGENDIR := $(TFMAKEFILE_DIR)/gen/protobuf_zuluko
PROTOBUFLIBS := $(PROTOBUFGENDIR)/lib/libprotobuf.a

# The location of nsync
NSYNCDIR := $(TFMAKEFILE_DIR)/downloads/nsync/builds/zuluko.linux.c++11
NSYNCLIBS := $(NSYNCDIR)/libnsync.a

OS_PATH :=
ifeq ($(HOST_OS),LINUX)
	OS_PATH=linux
endif
ifeq ($(HOST_OS),OSX)
	OS_PATH=darwin
endif
ifeq ($(HOST_OS),WINDOWS)
    $(error "windows is not supported.")
endif

ifndef STAGING_DIR
    $(error "STAGING_DIR is not defined.")
endif

CXX := $(STAGING_DIR)/bin/$(BIN_PREFIX)-g++
CC := $(STAGING_DIR)/bin/$(BIN_PREFIX)-gcc

CXXFLAGS +=\
-Wno-narrowing \
-fomit-frame-pointer

INCLUDES = \
-I. \
-I$(SCRIPT_DIR)/jpeg/include \
-I$(DOWNLOADSDIR) \
-I$(DOWNLOADSDIR)/eigen/ \
-I$(TFMAKEFILE_DIR)/downloads/nsync/public \
-I$(PROTOBUFGENDIR)/include/ \
-I$(PROTOGENDIR) \
-I$(PBTGENDIR)

LIBS := \
-ltensorflow-core \
-lprotobuf \
-lnsync \
-ljpeg \
-ldl \
-lm


LD := $(STAGING_DIR)/$(BIN_PREFIX)/bin/ld

LDFLAGS := \
$(MARCH_OPTION) \
-L$(TFLIBDIR) \
-L$(NSYNCDIR) \
-L$(PROTOBUFGENDIR)/lib \
-L$(SCRIPT_DIR)/jpeg/lib/zuluko \
-v

AR := $(STAGING_DIR)/bin/$(BIN_PREFIX)-ar
ARFLAGS := r
LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive

endif  # ZULUKO

# The name of the output program we're compiling.
EXECUTABLE_NAME := $(BINDIR)/image_recognizer_demo

EXECUTABLE_SRCS := tensorflow/contrib/pi_examples/input_image/image_recognizer_demo.cc
LIB_SRCS := tensorflow/contrib/pi_examples/input_image/image_recognizer.cc

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libpi_module_recognition.a
LIB_PATH := $(LIBDIR)$(LIB_NAME)

# File names of the intermediate files target compilation generates.
EXECUTABLE_OBJS := $(addprefix $(OBJDIR), $(EXECUTABLE_SRCS:.cc=.o))
LIB_OBJS := $(addprefix $(OBJDIR), $(LIB_SRCS:.cc=.o))

.PHONY: clean

LN_CMD := ln_cmd

# The target that's compiled if there's no command-line arguments.
all: $(LN_CMD) $(LIB_PATH) $(EXECUTABLE_NAME)

$(LN_CMD):
	ln -sf $(NSYNCDIR)/nsync.a $(NSYNCDIR)/libnsync.a

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

# Rules for target compilation.
$(EXECUTABLE_NAME): $(EXECUTABLE_OBJS) $(LIB_PATH) $(TFLIBS) $(PROTOBUFLIBS) $(NSYNCLIBS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(EXECUTABLE_NAME) $(EXECUTABLE_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

# Matches on C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Gets rid of all generated files.
clean:
	rm -rf $(OBJDIR)
	rm -rf $(LIBDIR)
	rm -rf $(BINDIR)
