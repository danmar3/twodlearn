# twodlearn
A library to develop machine learning models.

## A. Installation
* 1. Clone the project
```
git clone git@github.com:danmar3/twodlearn.git twodlearn
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

## B. Run the tests using pytest
install pytest
```
pip install -U pytest
```

run the unit-tests using pytest:
```
cd twodlearn/tests/
pytest -ra                # print a short test summary info at the end of the session
pytest -x --pdb           # drop to PDB on first failure, then end test session
pytest --pdb --maxfail=3  # drop to PDB for first three failures
pytest --durations=10     # get the test execution time
pytest --lf               # to only re-run the failures.
pytest --cache-clear      # clear the cache of failed tests
```
