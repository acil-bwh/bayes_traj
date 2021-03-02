"""https://packaging.python.org/en/latest/distributing.html"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bayes_traj',
    version='0.0.0',
    description='bayes_traj',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/acil-bwh/bayes_traj',
    author='James Ross',
    author_email='jross@bwh.harvard.edu',

    ### Other stuff ...
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    packages = find_packages(),
)
