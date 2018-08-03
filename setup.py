from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.1.6',
                     'h5py',
                     'pillow',
                     'tensorflow',
                     'numpy',
                     'scikit-learn',
                     'opencv-python']

setup(
    name='robocar pilot trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='ot trainer for robocar'
)
