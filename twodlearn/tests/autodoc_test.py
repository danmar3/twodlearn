import inspect
import unittest
import twodlearn as tdl


class DocstringTest(unittest.TestCase):
    def test_docstring(self):
        @tdl.core.create_init_docstring
        class TestLayer(tdl.core.Layer):
            @tdl.core.InputArgument
            def kernel_size(self, value):
                '''Size of the convolution kernels. Must be a tuple/list of two
                elements (height, width)
                '''
                return value

            @tdl.core.SimpleParameter
            def kernel(self, value):
                '''convolution kernel'''
                return value
        if tdl.core.PYTHON_VERSION >= 3:
            doc = inspect.getdoc(tdl.convnet.Conv2DLayer)
        else:
            doc = inspect.getdoc(tdl.convnet.Conv2DLayer.__init__)
        assert ("kernel_size (InputArgument): Size of the convolution "
                "kernels. Must be a tuple/list of two elements "
                "(height, width)") in doc


if __name__ == "__main__":
    unittest.main()
