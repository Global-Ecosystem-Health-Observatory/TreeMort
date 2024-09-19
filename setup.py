from setuptools import setup, find_packages

setup(
    name="TreeMort",
    version="0.1.0",
    author="@nis",
    license="GPLv3+",
    packages=find_packages(include=["treemort"]),
    install_requires=[
        'tqdm',
        'scipy',
        'torch',
        'matplotlib',
        'torchvision',
        'configargparse',
        'segmentation-models-pytorch',
        'opencv-python',
        'h5py',
        'rasterio',
        'transformers',
        'geojson',
        'geopandas',
    ],
)
