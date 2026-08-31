import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="patcid",
    version="1.0.0",
    author="Lucas Morin",
    author_email="lum@zurich.ibm.com",
    description="A Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DS4SD/PatCID",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    install_requires=[
        "ipykernel",
        "tqdm",
        "pandas",
        "rdkit",
        "mols2grid",
        "pdf2image",
        "matplotlib",
        "pillow"
    ],
    extras_require={
        # structure_finder: search a query structure inside your own documents.
        # See STRUCTURE_FINDER.md. Model-backed engines (MolGrapher, DECIMER,
        # DECIMER-Segmentation, MolClassifier) are installed separately.
        "structure-finder": [
            "pymupdf>=1.24",
            "numpy>=1.23",
            "opencv-python>=4.8",
            "python-docx>=1.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "structure-finder = structure_finder.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9', 
    package_data={"": ["*.json"]}
)