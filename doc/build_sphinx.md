see https://www.sphinx-doc.org/en/1.4/man/sphinx-apidoc.html
```
pip install Sphinx
pip install sphinx_rtd_theme
cd sphinx
sphinx-apidoc --separate --force -o source/ ../../twodlearn
make html
```
