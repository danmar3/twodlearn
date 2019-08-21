try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze
from setuptools import setup, find_packages
import pathlib
# for development installation: pip install -e .
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/twodlearn_version.tar.gz
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.rst").read_text()
# with open('README.md') as f:
#     README = f.read()


DEPS = ['pandas', 'pathlib', 'tqdm', 'matplotlib',
        'xarray', 'tensorflow-probability', 'tensorflow-datasets']


def get_dependencies():
    tf_names = ['tensorflow-gpu>=1.13,<2', 'tensorflow>=1.13,<2', 'tf-nightly']
    tf_installed = any([any(tfname == installed.split('==')[0]
                            for tfname in tf_names)
                        for installed in freeze.freeze()])
    if tf_installed:
        return DEPS
    else:
        return DEPS + ['tensorflow']


setup(name='twodlearn',
      version='0.5.0',
      description='Easy development of machine learning models',
      long_description=README,
      packages=find_packages(
          exclude=["*test*", "tests"]),
      # package_data={
      # '': ['*.h', '*.cu', 'makefile']
      # },
      package_data={'': ['*.so']},
      install_requires=get_dependencies(),
      python_requires='>=3.5.2',
      extras_require={
          'reinforce': ['gym', 'pybullet==2.4.5', 'casadi'],
          'development': ['pytest', 'line_profiler', 'pytest-faulthandler',
                          'jupyter'],
      },
      author='Daniel L. Marino',
      author_email='marinodl@vcu.edu',
      licence='Apache 2.0',
      url='https://github.com/danmar3/twodlearn'
      )
