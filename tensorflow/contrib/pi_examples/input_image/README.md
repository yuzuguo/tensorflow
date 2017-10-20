# building instruction

- *Must be in your repo root directory, run scripts below*

### Before you start (all platforms)
- download dependence
```bash
rm -rf tensorflow/contrib/makefile/downloads
tensorflow/contrib/makefile/download_dependencies.sh
```

### building on Linux
- dependence
```bash
sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev \
git python
sudo apt-get install -y libjpeg-dev
```
- building
```bash
tensorflow/contrib/makefile/build_all_linux.sh
```

- run benchmark for verification
  - run before download graph modes, if you have not.
  ```bash
  mkdir -p ~/graphs
  curl -o ~/graphs/inception.zip \
   https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip \
   && unzip ~/graphs/inception.zip -d ~/graphs/inception
  ```
  - run
  ```bash
  tensorflow/contrib/makefile/gen/bin/benchmark \
   --graph=$HOME/graphs/inception/tensorflow_inception_graph.pb
  ```
  
- building API static library and demo, after linux built on above steps
```bash
make -f tensorflow/contrib/pi_examples/input_image/Makefile clean
make -f tensorflow/contrib/pi_examples/input_image/Makefile
``` 

- run demo
```bash
tensorflow/contrib/pi_examples/input_image/gen/bin/image_classify_demo
```

### building on android

- building
```bash
make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID ANDROID_ARCH=arm64-v8a cleantarget

tensorflow/contrib/makefile/compile_android_protobuf.sh -c -a arm64-v8a

export HOST_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh`

export TARGET_NSYNC_LIB=`CC_PREFIX="${CC_PREFIX}" NDK_ROOT="${NDK_ROOT}" \
	tensorflow/contrib/makefile/compile_nsync.sh -t android -a arm64-v8a`

make -j12 -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID ANDROID_ARCH=arm64-v8a
```

- run benchmark on android
  - ignore steps
  
- building API static library and demo, after andorid built on above steps
```bash
make -f tensorflow/contrib/pi_examples/input_image/Makefile TARGET=ANDROID ANDROID_ARCH=arm64-v8a clean
make -f tensorflow/contrib/pi_examples/input_image/Makefile TARGET=ANDROID ANDROID_ARCH=arm64-v8a
```