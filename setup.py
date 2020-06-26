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
    scripts=['eval_scripts/mlr_eval_cv_coverage.py',
        'eval_scripts/mlr_eval_cv_klen.py',
        'eval_scripts/mlr_eval_cv_lambda.py',
        'eval_scripts/mlr_eval_cv_learning_rate.py',
        'report_scripts/plot_cv_klen.py'],
    install_requires=INSTALL_REQUIRES
)
