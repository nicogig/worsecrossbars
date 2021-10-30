from setuptools import setup


def load_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()

setup(
    name="worsecrossbars",
    version="1.1.0",
    packages=["worsecrossbars", "worsecrossbars.utilities",
              "worsecrossbars.backend", "worsecrossbars.plotting"],
    install_requires=load_requirements(),
    url="https://github.com/nicogig/worse-crossbars",
    license="MIT license",
    author="Lorenzo Bonito, Nicola Gigante, Yousef Mahmoud",
    author_email="lorenzo.bonito.18@ucl.ac.uk, nicola.gigante.18@ucl.ac.uk, yousef.mahmoud.18@ucl.ac.uk"
)
