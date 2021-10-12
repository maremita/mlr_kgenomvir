#!/usr/bin/env python

from mlr_kgenomvir.data.build_cv_data import build_load_save_cv_data
from mlr_kgenomvir.models.model_evaluation import perform_mlr_cv
from mlr_kgenomvir.models.model_evaluation import compile_score_names
from mlr_kgenomvir.models.model_evaluation import make_clf_score_dataframes
from mlr_kgenomvir.models.model_evaluation import plot_cv_figure
from mlr_kgenomvir.utils import str_to_list
from mlr_kgenomvir.utils import get_stats
from mlr_kgenomvir.utils import write_log

import sys
import configparser
import os.path
from os import makedirs
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, dump
from sklearn.base import clone

# Models to evaluate
from mlr_kgenomvir.models.pytorch_mlr import MLR
from sklearn.linear_model import LogisticRegression


__author__ = ["nicolas", "amine"]


"""
The script evaluates the performance of different 
regularized MLR models with function to low variance 
threshold of data for virus genome classification
"""


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Config file is missing!!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    # Get argument values from ini file
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(config_file, "r") as cf:
        config.read_file(cf)

    # virus
    virus_name = config.get("virus", "virus_code")

    # io
    seq_file = config.get("io", "seq_file")
    cls_file = config.get("io", "cls_file")
    outdir = config.get("io", "outdir")

    # seq_rep
    klen = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers")
    # ........ main evaluation parameters ..............
    lowVarThreshold = config.get("seq_rep", "low_var_threshold")
    # ..................................................
    fragmentSize = config.getint("seq_rep", "fragment_size",
            fallback=1000)
    fragmentCount = config.getint("seq_rep", "fragment_count",
            fallback=1000)
    fragmentCov = config.getfloat("seq_rep", "fragment_cov",
            fallback=2)

    # evaluation
    evalType = config.get("evaluation", "eval_type") # CC, CF or FF
    testSize = config.getfloat("evaluation", "test_size")
    cv_folds = config.getint("evaluation", "cv_folds")
    eval_metric = config.get("evaluation", "eval_metric")
    avrg_metric = config.get("evaluation", "avrg_metric")

    # classifier
    # sklearn or pytorch_mlr
    _module = config.get("classifier", "module") 
    _tol = config.getfloat("classifier", "tol")
    _lambda = config.getfloat("classifier", "lambda")
    _l1_ratio = config.getfloat("classifier", "l1_ratio")
    _solver = config.get("classifier", "solver")
    _max_iter = config.getint("classifier", "max_iter")
    _penalties = config.get("classifier", "penalty")

    if _module == "pytorch_mlr":
        _learning_rate = config.getfloat("classifier",
                "learning_rate")
        _n_iter_no_change = config.getint("classifier",
                "n_iter_no_change")
        _device = config.get("classifier", "device")

    # settings
    n_mainJobs = config.getint("settings", "n_main_jobs")
    n_cvJobs = config.getint("settings", "n_cv_jobs")
    verbose = config.getint("settings", "verbose",
            fallback=0)
    loadData = config.getboolean("settings", "load_data",
            fallback=False)
    saveData = config.getboolean("settings", "save_data",
            fallback=True)
    loadModels = config.getboolean("settings", "load_models",
            fallback=False)
    saveModels = config.getboolean("settings", "save_models",
            fallback=True)
    loadResults = config.getboolean("settings", "load_results",
            fallback=False)
    saveResults = config.getboolean("settings", "save_results",
            fallback=True)
    plotResults = config.getboolean("settings", "plot_results",
            fallback=True)
    randomState = config.getint("settings", "random_state",
            fallback=42)

    if evalType not in ["CC", "CF", "FF"]:
        raise ValueError(
                "evalType argument have to be one of CC, CF or"+
                " FF values")

    ## Tags for prefix out
    ######################
    if fullKmers:
        tag_kf = "F"
    else:
        tag_kf = "S"

    tag_fg = ""
    args_fg = dict()

    if evalType in ["CF", "FF"]:
        tag_fg = "FSZ{}_FCV{}_FCL{}_".format(str(fragmentSize),
                str(fragmentCov), str(fragmentCount))

        args_fg={'fragment_size':fragmentSize,
                'fragment_cov':fragmentCov,
                'fragment_count':fragmentCount}

    # OutDir folder
    ###############
    outdir = os.path.join(outdir,"{}/{}".format(virus_name,
        evalType))
    makedirs(outdir, mode=0o700, exist_ok=True)

    ## Thresholds to evaluate
    ########################
    thresholds_list = str_to_list(lowVarThreshold, cast=float)
    thresholds_list_str = [str(k) for k in thresholds_list]

    ## MLR initialization
    #####################
    if _module == "pytorch_mlr":
        mlr = MLR(tol=_tol, learning_rate=_learning_rate,
                l1_ratio=None, solver=_solver, max_iter=_max_iter,
                validation=False, n_iter_no_change=_n_iter_no_change,
                device=_device, random_state=randomState,
                verbose=verbose)
        mlr_name = "PTMLR"

    else:
        mlr = LogisticRegression(multi_class="multinomial", tol=_tol,
                solver=_solver, max_iter=_max_iter, verbose=0,
                l1_ratio=None)
        mlr_name = "SKMLR"

    ## Evaluate MLR models
    ######################
    # "l1", "l2", "elasticnet", "none"
    clf_penalties = str_to_list(_penalties)
    clf_names = [mlr_name+"_"+pen.upper() for pen in clf_penalties]

    clf_scores = defaultdict(dict)
    score_names = compile_score_names(eval_metric, avrg_metric)

    parallel = Parallel(n_jobs=n_mainJobs, prefer="processes",
            verbose=verbose)

    for ind, threshold in enumerate(thresholds_list):
        threshold_str = thresholds_list_str[ind]
        if verbose:
            print("\n{}. Evaluating variance threshold {}".format(
                ind+1, threshold_str), flush=True)

        # Construct prefix for output files
        ###################################
        prefix_out = os.path.join(outdir,
                "{}_{}_K{}{}_V{}_{}".format(virus_name,
                    evalType, tag_kf, klen, threshold_str, tag_fg))

        ## Generate training and testing data
        ####################################
        try:
            tt_data = build_load_save_cv_data(
                    seq_file,
                    cls_file,
                    prefix_out,
                    eval_type=evalType,
                    k=klen,
                    full_kmers=fullKmers,
                    low_var_threshold=threshold,
                    n_splits=cv_folds,
                    test_size=testSize,
                    load_data=loadData,
                    save_data=saveData,
                    random_state=randomState,
                    verbose=verbose,
                    **args_fg)
    
        except ValueError as e:
            print(e)
  
        cv_data = tt_data["data"]

        if verbose:
            print("X_train descriptive stats:\n{}".format(
                get_stats(cv_data["X_train"])))

        ## Train and compute performance of classifiers
        ###############################################
        # mlr_scores is a list of dictionaries of scores
        mlr_scores = parallel(delayed(perform_mlr_cv)(
            clone(mlr), clf_name, clf_penalty, _lambda,
            cv_data, prefix_out, metric=eval_metric,
            average_metric=avrg_metric, n_jobs=n_cvJobs,
            load_model=loadModels, save_model=saveModels,
            load_result=loadResults, save_result=saveResults,
            verbose=verbose, random_state=randomState)
            for clf_name, clf_penalty in zip(clf_names,
                clf_penalties))

        # Add the scores of current klen to clf_scores
        for i, clf_name in enumerate(clf_names):
            clf_scores[clf_name][threshold_str] = mlr_scores[i]

    # Rearrange clf_scores into dict of mean and std dataframes
    scores_dfs = make_clf_score_dataframes(clf_scores,
            thresholds_list_str, score_names, _max_iter)

    ## Save and Plot results
    ########################
    str_lr = ""
    if _module == "pytorch_mlr":
        str_lr = format(_learning_rate, '.0e')\
                if _learning_rate not in list(range(0, 10))\
                else str(_learning_rate)
        str_lr = "_LR"+str_lr

    str_lambda = format(_lambda, '.0e') if _lambda not in list(
            range(0, 10)) else str(_lambda)

    outFile = os.path.join(outdir,
            "{}_{}_K{}{}_V{}to{}_{}{}{}_A{}_LOWVARS_{}_{}".\
                    format(virus_name, evalType, tag_kf, klen,
                        thresholds_list_str[0],
                        thresholds_list_str[-1],
                        tag_fg, mlr_name, str_lr, str_lambda, 
                        avrg_metric, eval_metric))

    if saveResults:
        write_log(scores_dfs, config, outFile+".log")
        with open(outFile+".jb", 'wb') as fh:
            dump(scores_dfs, fh)

    if plotResults:
        plot_cv_figure(scores_dfs, score_names, thresholds_list_str,
                "Variance threshold", outFile)

    if verbose:
        print("\nFin normale du programme {}".format(sys.argv[0]))
