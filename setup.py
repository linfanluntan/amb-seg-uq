from setuptools import setup, find_packages

setup(
    name="amb-seg-uq",
    version="1.0.0",
    description="Ambiguity-Aware Uncertainty Quantification for Medical Image Segmentation",
    author="Renjie He",
    author_email="rhe@mdanderson.org",
    url="https://github.com/ambiguity-aware-uq/amb-seg-uq",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "nibabel>=5.0.0",
        "SimpleITK>=2.2.0",
        "nnunetv2>=2.3.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "h5py>=3.8.0",
        "medpy>=0.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
