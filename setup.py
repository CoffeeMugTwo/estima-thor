#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as fobj:
    install_requirements = [l.strip() for l in fobj.readlines()]

test_requirements = [ ]

setup(
    author="Dr. Thomas Kittler",
    author_email='thomas.kittler.01@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A simple tool for cost/time estimation of projects.",
    entry_points={
        'console_scripts': [
            'estimathor=estima_thor.cli:main',
        ],
    },
    install_requires=install_requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='estima_thor',
    name='estima_thor',
    packages=find_packages(include=['estima_thor', 'estima_thor.*']),
    project_urls={
        "Bug Tracker": "https://github.com/CoffeeMugTwo/estima-thor/issues"
    },
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/coffeemugtwo/estima_thor',
    version='0.1.0',
    zip_safe=False,
)
