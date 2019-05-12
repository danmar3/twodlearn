import os
import shutil
import unittest
import nbformat
import twodlearn
import twodlearn.common
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import PythonExporter

EXAMPLES_PATH = os.path.join(os.path.dirname(twodlearn.__file__),
                             'Examples')
TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


def convertNotebook(notebookPath, modulePath):
    with open(notebookPath) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    with open(modulePath, 'w+') as fh:
        fh.writelines(source)  # .encode('utf-8'))


def execute_notebook(notebook_path):
    assert os.path.exists(notebook_path),\
        'notebook {} does not exist'.format(notebook_path)
    print(' ------------- Running -------------- \n'
          '({})', notebook_path)
    with open(notebook_path) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)
    if twodlearn.common.PYTHON_VERSION == 2:
        ep = ExecutePreprocessor(timeout=600, kernel_name='python2')
    elif twodlearn.common.PYTHON_VERSION == 3:
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': TMP_PATH}})


class ExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(os.listdir(EXAMPLES_PATH))
        cls.tmp_path = TMP_PATH
        if os.path.exists(cls.tmp_path):
            shutil.rmtree(cls.tmp_path)
        os.makedirs(cls.tmp_path)

    def test_mlp(self):
        notebook_path = os.path.join(EXAMPLES_PATH,
                                     'simple_mlp.ipynb')
        execute_notebook(notebook_path)

    def test_convnet(self):
        notebook_path = os.path.join(EXAMPLES_PATH,
                                     'ConvnetMNIST.ipynb')
        execute_notebook(notebook_path)

    def test_bayesnet(self):
        notebook_path = os.path.join(EXAMPLES_PATH,
                                     'BayesDNN.ipynb')
        execute_notebook(notebook_path)

    def test_gp(self):
        notebook_path = os.path.join(EXAMPLES_PATH,
                                     'gaussian_process_1d.ipynb')
        execute_notebook(notebook_path)


if __name__ == "__main__":
    unittest.main()
