import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nets-abw",
    version="0.0.1",
    author="Andrew West",
    description="Network architectures written in TF 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abw-24/nets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)