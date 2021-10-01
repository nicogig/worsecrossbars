from setuptools import setup


def load_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='worsecrossbars',
    version='1.0.2',
    packages=['worsecrossbars', 'worsecrossbars.utilities',
              'worsecrossbars.backend', 'worsecrossbars.plotting'],
    install_requires=load_requirements(),
    url='https://github.com/nicogig/worse-crossbars',
    license='MIT license',
    author='Nicola Gigante, Lorenzo Bonito',
    author_email='zceengi@ucl.ac.uk'
)
