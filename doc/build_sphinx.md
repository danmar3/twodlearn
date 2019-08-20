pip install Sphinx
pip install sphinx_rtd_theme
cd sphinx
sphinx-apidoc -o source/ ../../twodlearn
make html
