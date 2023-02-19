from setuptools import find_packages, setup

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

__version__ = "0.0.1"

setup(
    name="longcatipcal",
    version=__version__,
    author="Chenglong Chen",
    author_email="c.chenglong@gmail.com",
    description="Long Capital",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=[],
    entry_points={},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Long Capital",
)
