sphinx-apidoc --separate --force -d 2 -o source/ ../../twodlearn
rm source/modules.rst
sphinx-build -E -a -b html source ../../../twodlearn-doc/
