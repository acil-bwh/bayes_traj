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
    version='0.0.2',
    description='bayes_traj',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/acil-bwh/bayes_traj',
    author='James Ross',
    author_email='jross@bwh.harvard.edu',

    scripts=[
        'bin/bayes_traj_main',
        'bin/viz_data_prior_draws',
        'bin/viz_model_trajs',        
        'bin/generate_generic_data',
        'bin/summarize_traj_model',
        'bin/generate_prior'],
    
    install_requires=[
        'provenance-tools >= 0.0.2',
        'pandas >= 1.1.1',
        'numpy >= 1.19.1'
    ],
    
    ### Other stuff ...
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    packages = find_packages(include=['bayes_traj_main',
                                      'viz_data_prior_draws',
                                      'viz_model_trajs',                                      
                                      'generate_generic_data',
                                      'generate_prior',
                                      'summarize_traj_model',
                                      'bin/*']),
)
