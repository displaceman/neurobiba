from setuptools import setup, find_packages

setup(
    name='neurobiba',
    version='0.6',
    description='small collection of functions for neural networks',
    url='https://github.com/displaceman/neurobiba',
    author_email='cumnaamys@gmail.com',
    author='displaceman',
    license='GPL',
    packages=find_packages(),
    install_requires=['numpy'],
    zip_safe=False)
