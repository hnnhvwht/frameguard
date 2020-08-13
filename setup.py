import pathlib
from setuptools import setup, find_packages


def read(relative_path):
    p = pathlib.Path(__file__).parent / relative_path
    with p.open() as f:
        return f.read()

def get_package_version(relative_path):
    contents = read(relative_path)
    for line in contents.splitlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
        else:
            raise RuntimeWarning("Failed to load version.")

setup(
    name="frameguard",
    version=get_package_version("frameguard/__init__.py"),
    description="Validated Pandas DataFrames",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    url="https://github.com/hnnhvwht/frameguard",
    keywords="pandas dataframe validation",
    project_urls={
        "Source": "https://github.com/hnnhvwht/frameguard",
    },
    author="Hannah White",
    author_email="hannah.white@tutanota.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.1",
        "pandas>=1.1.0",
        "pyyaml>=5.3.1",
    ],
)
