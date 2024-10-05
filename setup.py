from setuptools import setup, find_packages

setup(
    name="TreeMort",
    version="0.1.0",
    author="@nis",
    license="GPLv3+",
    packages=find_packages(include=["treemort"]),
    install_requires=[
        'tqdm',
        'h5py',
        'scipy',
        'torch',
        'geojson',
        'rasterio',
        'geopandas',
        'matplotlib',
        'torchvision',
        'transformers',
        'scikit-learn',
        'torchmetrics',
        'opencv-python',
        'configargparse',
        'segmentation-models-pytorch',
    ],
)
