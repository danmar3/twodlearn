# 2dlearn-lib_private

# A. Installation
* 1. Clone the project
```
git clone git@github.com:danmar3/2dlearn-lib.git twodlearn
cd twodlearn
```

* 2. Build the project only for deployment
```
bazel build -s --config=cuda_clang --verbose_failures --sandbox_debug //tools/pip_package:build_pip_package
bazel-bin/tools/pip_package/build_pip_package --src
```

* 3. Install the project
TODO


# B. Installation (Minimal)
* 1. Clone the project
```
git clone git@github.com:danmar3/2dlearn-lib.git twodlearn
cd twodlearn
```

* 2. Install the project
```
pip install -e .
```

## C. Installation (For development)
* 1. Clone the project
```
git clone git@github.com:danmar3/2dlearn-lib.git twodlearn
cd twodlearn
```

* 2. Build all targets (for development)
```
bazel build -s --config=cuda_clang --verbose_failures --sandbox_debug //...
bazel-bin/tools/pip_package/build_pip_package --src twodlearn_tmp
```

* 3. Install the project
```
cd twodlearn_tmp
pip install -e .
```

## D. Using nosetest
run the unittests using nose:
```
nosetests -v --with-id   # this keeps track of failed tests
nosetests -v --failed    # run only failed tests
nosetests -v --with-id   # reset the tracking of failed tests
```
using rednose:
```
nosetests --rednose -v --with-id
```
