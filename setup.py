import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

with open(os.path.join('textwiser', '_version.py')) as fp:
    exec(fp.read())

setuptools.setup(
    name="textwiser",
    description="TextWiser: Text Featurization Library",
    long_description=long_description,
    version=__version__,
    author="FMR LLC",
    url="https://",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    # install_requires=required,
    python_requires=">=3.6",
    classifiers=[
        "License :: FMR LLC",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "link"
    }
)
