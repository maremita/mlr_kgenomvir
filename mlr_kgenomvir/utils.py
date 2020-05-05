import os.path
import json
from pprint import pprint
from collections import defaultdict
import platform
import importlib

import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import matplotlib.cm as cm


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

def compile_score_names(eval_metric, avrg_metric):
    names = []
    if eval_metric == "all":
        names = ["test_{}_precision".format(avrg_metric), 
                "test_{}_recall".format(avrg_metric), 
                "test_{}_fscore".format(avrg_metric)]
    else:
        names = ["train_{}_{}".format(avrg_metric, eval_metric),
                "test_{}_{}".format(avrg_metric, eval_metric)]

    names.extend(["coef_sparsity", "convergence", "X_test_sparsity"])

    return names


def make_clf_score_dataframes(clf_covs, rows, columns, max_iter):
    # rows are coverages_str 
    # columns are score_names 
    df_scores = defaultdict(dict)

    for clf_name in clf_covs:
        df_mean = pd.DataFrame(index=rows, columns=columns)
        df_std = pd.DataFrame(index=rows, columns=columns)

        for row in rows:
            scores = clf_covs[clf_name][row]
            for score_name in columns:
                if score_name != "convergence":
                    df_mean.loc[row, score_name] = scores[score_name][0]
                    df_std.loc[row, score_name] = scores[score_name][1]
                else:
                    df_mean.loc[row,score_name] = scores["n_iter"][0]/max_iter
                    df_std.loc[row, score_name] = scores["n_iter"][1]/max_iter

        df_scores[clf_name]["mean"] = df_mean.astype(np.float)
        df_scores[clf_name]["std"] = df_std.astype(np.float)

    return df_scores


def plot_cv_figure(scores, score_labels, x_values, xlabel,  out_file):
    fig_format = "png"
    #fig_format = "eps"
    fig_dpi = 150

    fig_file = out_file+"."+fig_format
    fig_title = os.path.splitext((os.path.basename(fig_file)))[0]
    
    nb_clfs = len(scores)

    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/20) for j in range(0,20)]

    styles = ["s-","o-","d-.","^-.","x-","h-","<-",">-","*-","p-"]
    sizefont = 12

    f, axs = plt.subplots(1, nb_clfs, figsize=(8*nb_clfs, 5))

    plt.rcParams.update({'font.size':sizefont})
    plt.subplots_adjust(wspace=0.12, hspace=0.1)

    line_scores = [l for l in score_labels if "X" not in l]
    area_scores = [l for l in score_labels if "X" in l]    

    ind = 0
    for i_c, classifier in enumerate(scores):
        df_mean = scores[classifier]["mean"]
        df_std = scores[classifier]["std"]

        dfl_mean = df_mean[line_scores]
        dfl_std = df_std[line_scores]

        dfa_mean = df_mean[area_scores]
        dfa_std = df_std[area_scores]

        p = dfl_mean.plot(kind='line', ax=axs[ind], style=styles, 
                fontsize=sizefont, markersize=8)

        dfa_mean.plot(kind='area', ax=axs[ind], alpha=0.2, color=colors,
                fontsize=sizefont)

        # For ESP transparent rendering
        p.set_rasterization_zorder(0)

        xticks = [j for j in range(len(x_values))]
        xticks = np.array(xticks)
 
        p.set_title(classifier)
        p.set_xticks(xticks)
        p.set_xticklabels(x_values, fontsize=sizefont)
        p.set_ylim([-0.05, 1.05])
        p.set_xlabel(xlabel, fontsize=sizefont+1) # 'Coverage'

        zo = -ind
        for score_name in dfl_mean:
            m = dfl_mean[score_name]
            s = dfl_std[score_name]

            p.fill_between(xticks, m-s, m+s, alpha=0.1, zorder=zo)
            zo -= 1

        p.get_legend().remove()
        p.grid()
        ind += 1
    
    # print legend for the last subplot
    p.legend(loc='upper left', fancybox=True, shadow=True, 
            bbox_to_anchor=(1.01, 1.02))

    plt.suptitle(fig_title)
    plt.savefig(fig_file, bbox_inches="tight",
            format=fig_format, dpi=fig_dpi)


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
