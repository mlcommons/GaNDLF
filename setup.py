#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
  readme = readme_file.read()

requirements = [
  'numpy',
  'scipy',
  'SimpleITK==1.2.4',
  'torch>=1.6',
  'torchvision',
  'tqdm',
  'torchio==0.17.50',
  'pandas',
  'pylint',
  'torchsummary',
  'scikit-learn==0.23.1',
  'pickle5==0.0.11',
  'setuptools',
  'seaborn',
  'pyyaml',
  'openslide-python',
  'scikit-image',
  'matplotlib'
]

setup(
  name='GANDLF',
  version='0.0.5.NR', # NR: non-release; this should be changed when tagging\
  author="Megh Bhalerao, Caleb Grenko, Siddhesh Thakur, Sarthak Pati",
  author_email='software@cbica.upenn.edu',
  python_requires='>=3.6',
  scripts=['gandlf_run', 'gandlf_constructCSV', 'gandlf_collectStats', 'gandlf_patchMiner'],
  classifiers=[
    'Development Status :: Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD-3-Clause License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
  description=(
    "Semantic segmentation using various DL architectures"
    " on PyTorch."
  ),
  install_requires=requirements,
  license="BSD-3-Clause License",
  long_description=readme,
  long_description_content_type='text/markdown',
  include_package_data=True,
  keywords='semantic, segmentation, brain, breast, liver, lung, augmentation',
  zip_safe=False,
)
