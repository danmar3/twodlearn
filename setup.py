try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze
from setuptools import setup, find_packages
# for development installation: pip install -e .
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/twodlearn_version.tar.gz
DEPS = ['tensorflow-gpu', 'pandas', 'pathlib', 'tqdm', 'matplotlib',
        'xarray']


def get_dependencies():
    if any(['tensorflow' in installed for installed in freeze.freeze()]):
        return [dep for dep in DEPS if 'tensorflow' not in dep]
    else:
        return DEPS


setup(name='twodlearn',
      version='0.3.1',
      packages=find_packages(
          exclude=["*test*", "tests"]),
      # package_data={
      # '': ['*.h', '*.cu', 'makefile']
      # },
      package_data={'': ['*.so']},
      install_requires=get_dependencies(),
      extras_require={
          'reinforce': ['gym==0.10.2', 'pybullet==1.9.2'],
          'development': ['nose', 'nose-timer', 'tensorflow-probability-gpu'],
      },
      author='Daniel L. Marino',
      author_email='marinodl@vcu.edu',
      licence='GPL',
      url='https://github.com/danmar3/2dlearn-lib'
      )
