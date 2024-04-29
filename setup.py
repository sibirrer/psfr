#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Simon Birrer",
    author_email='sibirrer@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.11',
    ],
    description="Telescope Images Point Spread Function Reconstruction",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    license = 'BSD 3-Clause',
    license_file = 'LICENSE.rst',
    keywords='psfr',
    name='psfr',
    packages=find_packages(include=['psfr', 'psfr.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sibirrer/psfr',
    version='0.1.0',
    zip_safe=False,
)
