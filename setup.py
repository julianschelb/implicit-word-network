import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="implicit-word-network",
    version="0.0.3",
    author="Julian Schelb",
    author_email="julian.schelb@uni-konstanz.de",
    description="A python package for extracting and exploring context-enriched word networks from corpora",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inf.uni-konstanz.de/julian.schelb/implicit-word-network",
    project_urls={
        "Bug Tracker": "https://gitlab.inf.uni-konstanz.de/julian.schelb/implicit-word-network/-/issues",
    },
    install_requires=[  # I get to this in a second
        "sklearn",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
