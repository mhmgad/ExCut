from setuptools import setup, find_packages
from ampligraph import __version__ as version

setup_params = dict(name='ampligraph',
                    version=version,
                    description='ExCut: A Python library for Explainable Entity Clustering',
                    url='https://github.com/mhmgad/ExCut',
                    author='Mohamed Gad-Elrab',
                    author_email='gadelrab@mpi-inf.mpg.de',
                    license='Apache 2.0',
                    packages=find_packages(exclude=('tests', 'docs')),
                    include_package_data=True,
                    zip_safe=False,
                    install_requires=[
                      "scikit-learn",
                      "tqdm",
                      "pandas",
                      "sparqlwrapper",
                      "rdflib=4.2.2",
                      "matplotlib",
                      "tensorflow=1.13.1",
                      "keras",
                      "ampligraph==1.3.0"
                    ])
