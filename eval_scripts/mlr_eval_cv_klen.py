#!/usr/bin/env python

from mlr_kgenomvir.data.build_cv_data import build_load_save_cv_data
from mlr_kgenomvir.models.model_evaluation import perform_mlr_cv
from mlr_kgenomvir.models.model_evaluation import compile_score_names
from mlr_kgenomvir.models.model_evaluation import make_clf_score_dataframes
from mlr_kgenomvir.models.model_evaluation import plot_cv_figure
from mlr_kgenomvir.utils import str_to_list
from mlr_kgenomvir.utils import write_log

import sys
import configparser
import os.path
from os import makedirs
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import clone

# Models to evaluate
from mlr_kgenomvir.models.pytorch_mlr import MLR
from sklearn.linear_model import LogisticRegression


__author__ = "amine"


"""
The script evaluates the performance of different regularized MLR models
with function to kmer lengths for virus genome classification
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
    # ........ main evaluation parameters ..............
    k_lenghts = config.get("seq_rep", "k")
    # ..................................................
    fullKmers = config.getboolean("seq_rep", "full_kmers") 

    # evaluation
    evalType = config.get("evaluation", "eval_type") # CF or FF only
    testSize = config.getfloat("evaluation", "test_size") 
    cv_folds = config.getint("evaluation", "cv_folds") 
    eval_metric = config.get("evaluation", "eval_metric")
    avrg_metric = config.get("evaluation", "avrg_metric")

    # classifier
    _module = config.get("classifier", "module") # sklearn or pytorch_mlr
    _tol = config.getfloat("classifier", "tol")
    _lambda = config.getfloat("classifier", "lambda") 
    _l1_ratio = config.getfloat("classifier", "l1_ratio") 
    _solver = config.get("classifier", "solver")
    _max_iter = config.getint("classifier", "max_iter")
    _penalties = config.get("classifier", "penalty")

    if _module == "pytorch_mlr":
        _learning_rate = config.getfloat("classifier", "learning_rate")
        _n_iter_no_change = config.getint("classifier", "n_iter_no_change")
        _device = config.get("classifier", "device")

    # settings 
    nJobs = config.getint("settings", "n_jobs")
    verbose = config.getint("settings", "verbose")
    saveFiles = config.getboolean("settings", "save_files")
    randomState = config.getint("settings", "random_state")

    if evalType in ["CC", "CF", "FF"]:
        if evalType in ["CF", "FF"]:
            try:
                fragmentSize = config.getint("seq_rep", "fragment_size")
                fragmentCount = config.getint("seq_rep", "fragment_count")
                fragmentCov = config.getfloat("seq_rep", "fragment_cov") 

            except configparser.NoOptionError:
                raise configparser.NoOptionError()
    else:
        raise ValueError("evalType argument have to be one of CC, CF or"+
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
    outdir = os.path.join(outdir,"{}/{}".format(virus_name, evalType))
    makedirs(outdir, mode=0o700, exist_ok=True)

    ## K lengths to evaluate
    ########################
    klen_list = str_to_list(k_lenghts, cast=int)
    klen_list_str = [str(k) for k in klen_list]

    ## MLR initialization
    #####################

    if _module == "pytorch_mlr":
        mlr = MLR(tol=_tol, learning_rate=_learning_rate, l1_ratio=None, 
                solver=_solver, max_iter=_max_iter, validation=False, 
                n_iter_no_change=_n_iter_no_change, device=_device, 
                random_state=randomState, verbose=verbose)
        mlr_name = "PTMLR"

    else:
        mlr = LogisticRegression(multi_class="multinomial", tol=_tol, 
                solver=_solver, max_iter=_max_iter, verbose=0, l1_ratio=None)
        mlr_name = "SKMLR"

    ## Evaluate MLR models
    ######################

    # "l1", "l2", "elasticnet", "none"
    clf_penalties = str_to_list(_penalties)
    clf_names = [mlr_name+"_"+pen.upper() for pen in clf_penalties]

    clf_scores = defaultdict(dict)
    score_names = compile_score_names(eval_metric, avrg_metric)

    parallel = Parallel(n_jobs=nJobs, prefer="processes", verbose=verbose)

    # If we have enough memory we can parallelize this loop
    for klen in klen_list:
        if verbose:
            print("\nEvaluating K {}".format(klen), flush=True)

        # Construct prefix for output files
        ###################################
        prefix_out = os.path.join(outdir, "{}_{}_K{}{}_{}".format(
            virus_name, evalType, tag_kf, klen, tag_fg))

        ## Generate training and testing data
        ####################################
        tt_data = build_load_save_cv_data(
                seq_file,
                cls_file,
                prefix_out,
                eval_type=evalType,
                k=klen,
                full_kmers=fullKmers, 
                n_splits=cv_folds,
                test_size=testSize,
                random_state=randomState,
                verbose=verbose,
                **args_fg)

        cv_data = tt_data["data"]

        mlr_scores = parallel(delayed(perform_mlr_cv)(clone(mlr), clf_name,
            clf_penalty, _lambda, cv_data, prefix_out, metric=eval_metric,
            average_metric=avrg_metric, n_jobs=cv_folds, save_files=saveFiles,
            verbose=verbose, random_state=randomState)
            for clf_name, clf_penalty in zip(clf_names, clf_penalties))
        #print(mlr_scores)

        for i, clf_name in enumerate(clf_names):
            clf_scores[clf_name][str(klen)] = mlr_scores[i]
 
    #print(clf_scores)

    scores_dfs = make_clf_score_dataframes(clf_scores, klen_list_str, 
            score_names, _max_iter)

    #pprint(scores_dfs)

    ## Save and Plot results
    ########################
    outFile = os.path.join(outdir, 
            "{}_{}_K{}{}to{}_{}A{}_{}_KLENGTHS".format(virus_name, evalType,
                tag_kf, klen_list[0], klen_list[-1], tag_fg, _lambda,
                mlr_name ))

    if saveFiles:
        write_log(scores_dfs, config, outFile+".log")

    plot_cv_figure(scores_dfs, score_names, klen_list_str, "K length", 
            outFile)
