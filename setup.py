"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "elm", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")


with open("requirements.txt") as f:
    install_requires = f.readlines()


test_requires = ["pytest>=5.2", "pytest-mock", "pytest-asyncio", "pytest-cov"]
description = "Energy Language Model"

setup(
    name="NREL-elm",
    version=version,
    description=description,
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.com/NREL/elm",
    packages=find_packages(),
    package_dir={"elm": "elm"},
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="elm",
    python_requires='>=3.9',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=install_requires,
    extras_require={
        "dev": install_requires + test_requires,
    },
    entry_points={"console_scripts": ["elm=elm.cli:main"]}
)
