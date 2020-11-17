from setuptools import setup

requirements = [
    'openslide-python',
    'scikit-image',
    'matplotlib',
    'numpy',
    'tqdm',
    'pandas'
]

setup(
    name='OPM-GANDLF',
    version='0.1.0',
    packages=['opm'],
    install_requires=requirements,
    url='https://github.com/grenkoca/OPM-GANDLF',
    license='BSD-3-Clause License',
    author='Caleb Grenko',
    author_email='grenkoca@gmail.com',
    description='A patch miner for GANDLF'
)
