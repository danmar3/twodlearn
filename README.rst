twodlearn
=========

A library to develop machine learning models.

A. Installation
---------------

* 1. Install the desired version of tensorflow (CPU or GPU)
  ::

    pip install tensorflow        # for CPU
    pip install tensorflow-gpu    # for GPU


* 2. Clone the project
  ::

    git clone git@github.com:danmar3/twodlearn.git twodlearn
    cd twodlearn


* 3. Install the project
  ::

    pip install -e .

* 4. Install extras (optional)
  ::

    pip install -e .[reinforce]
    pip install -e .[development]


B. Run the tests using pytest
-----------------------------
install pytest ``pip install -U pytest``

run the unit-tests using pytest:
::

  cd twodlearn/tests/
  pytest -ra                # print a short test summary info at the end of the session
  pytest -x --pdb           # drop to PDB on first failure, then end test session
  pytest --pdb --maxfail=3  # drop to PDB for first three failures
  pytest --durations=10     # get the test execution time
  pytest --lf               # to only re-run the failures.
  pytest --cache-clear      # clear the cache of failed tests



Roadmap for v0.6
----------------
- [x] migrate to TF 1.14
- [ ] add documentation
- [ ] add project to pypi
- [ ] create LayerNamespace
- [x] add a shortcut for required and optional input arguments
- [x] add check_arguments method to Layer and TdlModel
- [x] get_parameters now supports nested structures and nested SimpleNamespace
- [ ] deprecate tuple initialization
- [ ] deprecate optim
- [ ] move feedforward to dense
- [ ] cleanup common: clean deprecated descriptors and put them in separate file
- [ ] remove redundant base classes, such as TdlObject
- [ ] deprecate templates and design a format for estimators
- [ ] deprecate options value
- [ ] deprecate pyfmi and jmodelica
