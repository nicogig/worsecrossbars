"""Setup module.
Called by pip when installing the package.
"""
from typing import List

from setuptools import setup


def load_requirements() -> List[str]:
    """Load the requirements from the associated file."""
    with open("requirements.txt", encoding="utf8") as file:
        return file.read().splitlines()


setup(
    name="worsecrossbars",
    version="1.3.1",
    packages=[
        "worsecrossbars",
        "worsecrossbars.utilities",
        "worsecrossbars.backend",
        "worsecrossbars.plotting",
    ],
    install_requires=load_requirements(),
    url="https://github.com/nicogig/worse-crossbars",
    license="MIT license",
    author="Lorenzo Bonito, Nicola Gigante, Yousef Mahmoud",
    author_email="lorenzo.bonito.18@ucl.ac.uk,"
    + "nicola.gigante.18@ucl.ac.uk, yousef.mahmoud.18@ucl.ac.uk",
)
