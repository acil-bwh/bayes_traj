"""https://packaging.python.org/en/latest/distributing.html"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path
import os, re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

#packages = find_packages(include=['bayes_traj_main',
#                                  'viz_data_prior_draws',
#                                  'viz_model_trajs',                                      
#                                  'generate_generic_data',
#                                  'generate_prior',
#                                  'summarize_traj_model',
#                                  'bin/*'])

packages = find_packages()
packages.append('bayes_traj')

# Read version from the package __init__.py file
def get_version():
    init_path = os.path.join(here, "bayes_traj", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
            if match:
                return match.group(1)
    raise RuntimeError("Unable to find version string in bayes_traj/__init__.py")


setup(
    name='bayes_traj',
    version=get_version(),
    description='bayes_traj',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/acil-bwh/bayes_traj',
    author='James Ross',
    author_email='jross@bwh.harvard.edu',

    entry_points = {"console_scripts": ['bayes_traj_main = bayes_traj.bayes_traj_main:main',
                                        'viz_data_prior_draws = bayes_traj.viz_data_prior_draws:main',
                                        'viz_model_trajs = bayes_traj.viz_model_trajs:main',
                                        'viz_gamma_dists = bayes_traj.viz_gamma_dists:main',
                                        'generate_generic_data = bayes_traj.generate_generic_data:main',
                                        'summarize_traj_model = bayes_traj.summarize_traj_model:main',
                                        'assign_trajectory = bayes_traj.assign_trajectory:main',
                                        'update_model = bayes_traj.update_model:main',
                                        'get_alpha_estimate = bayes_traj.get_alpha_estimate:main',
                                        'generate_prior = bayes_traj.generate_prior:main']},
    
    install_requires=[
        'provenance-tools >= 0.0.5',
        'pandas < 2.2.2',
        'numpy >= 1.26.4',
        'matplotlib >= 3.3.1',
        'scipy >= 1.5.2',
        'argparse >= 1.1',
        'statsmodels >= 0.11.1',
        'torch >= 2.0.1',
        'pyro-ppl >= 1.8.5',
        'pytest >= 7.0.0',
        'numexpr >= 2.10.0',
        'bottleneck >= 1.3.8',        
    ],
    
    
    ### Other stuff ...
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    packages=packages,
)
