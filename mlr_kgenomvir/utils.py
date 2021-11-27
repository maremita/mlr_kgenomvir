import json
from pprint import pprint
import platform
import importlib

import numpy as np
import scipy
import scipy.sparse as sp
import configparser

import pandas as pd

__author__ = "amine"


def read_config_file(conf_file):

    cfg = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(conf_file, "r") as cf:
        cfg.read_file(cf)

    return cfg


def load_Xy_data(xfile, yfile):

    X = scipy.sparse.load_npz(xfile)
    with open(yfile, 'r') as fh: y = np.array(json.load(fh))

    return X, y


def save_Xy_data(X, xfile, y, yfile):

    scipy.sparse.save_npz(xfile, X)
    with open(yfile, 'w') as fh: json.dump(y.tolist(), fh)


def load_Xy_cv_data(cv_file):
    with np.load(cv_file, allow_pickle=True) as f:
        return f['data'].tolist()


def save_Xy_cv_data(data, cv_file):
    np.savez(cv_file, data=data)


def get_module_versions():
    versions = dict()

    versions["python"] = platform.python_version()

    module_names = ["mlr_kgenomvir", "numpy", "scipy", "pandas",
            "sklearn", "Bio", "joblib", "matplotlib", "torch",
            "lxml", "dendropy", "phylodm"]

    for module_name in module_names:
        found = importlib.util.find_spec(module_name)
        if found:
            module = importlib.import_module(module_name)
            versions[module_name] = module.__version__
        else:
            versions[module_name] = "Not found"

    return versions


# Evaluation scripts
# ##################

def str_to_list(chaine, sep=",", cast=None):
    c = lambda x: x
    if cast: c = cast

    return [c(i.strip()) for i in chaine.split(sep)]


def write_log(results, args, out_file):

    with open(out_file, "wt") as f:
        with pd.option_context('display.max_rows', None, 
                'display.max_columns', None):
            f.write("Final results\n##############\n\n")
            pprint(results, stream=f)

        f.write("\nProgram arguments\n################\n\n")
        args.write(f)

        f.write("Package versions\n################\n\n")
        pprint(get_module_versions(), stream=f)


# Descriptive stats of np matrix
def get_stats(mat):
    chaine = "It is not numpy.ndarray. Stats wont be calculated\n"

    if isinstance(mat, np.ndarray) or sp.issparse(mat):
        chaine = "\tShape {}\n".format(mat.shape)
        #approximation
        chaine += "\tMemory ~{} MB\n".format(mat.nbytes/1e6)

        if sp.issparse(mat):
            chaine += "\tSparsity {}\n".format(
                    np.mean(mat.todense().ravel() == 0))
        else:
            chaine += "\tSparsity {}\n".format(
                    np.mean(mat.ravel() == 0))

        feat_vars = mat.var(axis=0)
        chaine += "\tMean of feature variances {}\n".format(
                feat_vars.mean())
        chaine += "\tVariance of feature variances {}\n".format(
                feat_vars.var())
        #chaine += "\tFlags\n{}\n".format(mat.flags)

    return chaine
