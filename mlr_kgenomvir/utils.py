import json
from pprint import pprint
import platform
import importlib

import numpy as np
import scipy


__author__ = "amine"


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

    module_names = ["python", "mlr_kgenomvir", "numpy", "scipy", "pandas", 
            "sklearn", "Bio", "joblib", "matplotlib", "torch"]

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
        f.write("Final results\n##############\n\n")
        pprint(results, stream=f)

        f.write("\nProgram arguments\n################\n\n")
        args.write(f)

        f.write("Package versions\n################\n\n")
        pprint(get_module_versions(), stream=f)
