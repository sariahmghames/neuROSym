#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['neurosym_sgan'],
    package_dir={'neurosym_sgan': 'src/neuROSym/motion_predict/neurosym_sgan'}
)

setup(**setup_args)
