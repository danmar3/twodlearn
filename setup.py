try:
    from pip._internal.operations import freeze
    from pip._internal.exceptions import InstallationError
except ImportError:  # pip < 10.0
    from pip.operations import freeze
    from pip.exceptions import InstallationError
import operator
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


def split_name_ver(input):
    '''split string in name and version'''
    sep = min(input.find(op) if input.find(op) > 0 else len(input)
              for op in ('=', '>', '<'))
    return input[:sep], input[sep:]


def check_dep(installed, target):
    '''check if installed '''
    def split_ver_op(input):
        '''split string in version and operation'''
        ops = {'==': operator.eq, '>=': operator.ge, '<=': operator.le,
               '>': operator.gt, '<': operator.lt}
        if input[:2] in ops:
            op, ver = ops[input[:2]], input[2:]
        elif input[:1] in ops:
            op, ver = ops[input[:1]], input[1:]
        else:
            raise ValueError('version {} not valid'.format(input))
        return ver, op

    def check_ver(installed, target, op):
        '''check version for a given operation'''
        target = (int(i) for i in target.split('.')
                  if i.isdigit())
        installed = (int(i) for i in installed.split('.')
                     if i.isdigit())
        return all(op(ins, tar)
                   for ins, tar in zip(installed, target))

    target_name, target_ver = split_name_ver(target)
    installed_name, installed_ver = installed.split('==')

    if target_name != installed_name:
        return False
    if not target_ver:
        return True
    ver = target_ver.split(',')
    if not check_ver(installed_ver, *split_ver_op(ver[0])):
        return False
    if len(ver) == 1:
        return True
    return check_ver(installed_ver, *split_ver_op(ver[1]))


def get_dependencies():
    tf_names = ['tensorflow-gpu>=1.14,<2', 'tensorflow>=1.14,<2', 'tf-nightly']
    tf_installed = any([
        any(installed.split('==')[0] == split_name_ver(tfname)[0]
            for tfname in tf_names)
        for installed in freeze.freeze()])
    if tf_installed:
        tf_check = any([
            any(check_dep(installed=installed, target=tfname)
                for tfname in tf_names)
            for installed in freeze.freeze()
            if '==' in installed])
        if tf_check:
            return DEPS
        else:
            raise InstallationError(
                'Unsupported version of tensorflow is installed. '
                'Supported versions are {}'.format(tf_names))
    else:
        return DEPS + ['tensorflow']


setup(name='twodlearn',
      version='0.6.0',
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
