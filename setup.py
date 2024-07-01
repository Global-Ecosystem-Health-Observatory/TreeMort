from setuptools import setup

setup(
    name="treemortality",
    version="0.1.0",
    author="@nis",
    license="GPLv3+",
    install_requires=[
        "tensorflow[and-cuda]",
        "configargparse",
        "tqdm",
        "scipy",
        "matplotlib",
    ],
)
