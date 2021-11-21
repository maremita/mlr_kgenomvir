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
    author_email='remita.amine@courrier.uqam.ca',
    packages=find_packages(),
    package_data={'': ['*.jar', 'mlr_kgenomvir/simulation/santa.jar']},
    include_packages_data=True,
    scripts=[
        #
        'eval_scripts/mlr_eval_cv_coverage.py',
        'eval_scripts/mlr_eval_cv_klen.py',
        'eval_scripts/mlr_eval_cv_lambda.py',
        'eval_scripts/mlr_eval_cv_learning_rate.py',
        'eval_scripts/mlr_eval_cv_lowVar.py',
        #
        'eval_scripts/mlr_eval_sim_coverage.py',
        'eval_scripts/mlr_eval_sim_evoparams.py',
        'eval_scripts/mlr_eval_sim_imbalanced_dataset.py',
        'eval_scripts/mlr_eval_sim_imbalanced_sampling.py',
        'eval_scripts/mlr_eval_sim_klen.py',
        'eval_scripts/mlr_eval_sim_lambda.py',
        'eval_scripts/mlr_eval_sim_learning_rate.py',
        'eval_scripts/mlr_eval_sim_lowVar.py',
        'eval_scripts/mlr_eval_sim_nbclasses.py',
        #
        'hpc_scripts/write_run_mlr_sim_exps.py',
        'hpc_scripts/submit_mlr_exp.sh',
        #
        'report_scripts/mlr_plot_cv_klen.py',
        'report_scripts/mlr_plot_cv_lambda.py',
        'report_scripts/mlr_plot_cv_learning_rate.py'],
    install_requires=INSTALL_REQUIRES
)
