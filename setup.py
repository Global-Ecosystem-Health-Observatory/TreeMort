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
        'matplotlib',
        'tensorflow',
        'configargparse',
        'segmentation_models',
    ],
)
