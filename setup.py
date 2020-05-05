from setuptools import setup, find_packages
from mlr_kgenomvir import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='mlr_kgenomvir',
    version=_version,
    description='Evaluation of multinomial logistic regression'+\
            ' for kmer-based genome virus classification',
    author='remita',
    author_email='amine.m.remita@gmail.com',
    packages=find_packages(),
    #scripts=['scripts/mlrgv_eval_holdout_coverage.py',
    #    'scripts/mlrgv_eval_holdout_k.py',
    #    'scripts/mlrgv_eval_holdout_lambda.py'],
    install_requires=INSTALL_REQUIRES
)
