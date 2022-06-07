import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

with open("requirements_full.txt") as fh:
    full_reqs = fh.read().splitlines()

with open(os.path.join('textwiser', '_version.py')) as fp:
    exec(fp.read())

setuptools.setup(
    name="textwiser",
    description="TextWiser: Text Featurization Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    author="FMR LLC",
    url="https://github.com/fidelity/textwiser",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=required,
    python_requires=">=3.6",
    extras_require={"full": full_reqs},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/fidelity/textwiser",
        "Documentation": "https://fidelity.github.io/textwiser/"
    }
)
