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

* 3. Install extras (optional)
```
pip install -e .[reinforce]
pip install -e .[development]
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

### D.1 Using pytest (recommended)
install pytest
```
pip install -U pytest
```

run the unittests using pytest:
```
pytest -ra                # print a short test summary info at the end of the session
pytest -x --pdb           # drop to PDB on first failure, then end test session
pytest --pdb --maxfail=3  # drop to PDB for first three failures
pytest --durations=10     # get the test execution time
pytest --lf               # to only re-run the failures.
pytest --cache-clear      # clear the cache of failed tests
```

## D.2 Using nosetest
install nose
```
pip install nose nose-timer rednose
```
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
using nose-timer
```
nosetests --with-timer
```
using profile:
```
nosetests --with-profile --rednose -v --with-id
```
