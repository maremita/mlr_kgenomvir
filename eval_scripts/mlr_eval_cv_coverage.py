#!/usr/bin/env python

from mlr_kgenomvir.data.build_cv_data import build_load_save_cv_data
from mlr_kgenomvir.models.model_evaluation import perform_sk_mlr_cv
from mlr_kgenomvir.utils import compile_score_names
from mlr_kgenomvir.utils import make_clf_score_dataframes
from mlr_kgenomvir.utils import plot_cv_figure
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
from sklearn.linear_model import LogisticRegression


__author__ = "amine"


"""
The script evaluates the effect of the coverage of fragments on genome 
positions on the performance of different regularized MLR models for virus
genome classification
"""


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Config file is missing!!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    # Get argument values from ini file
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    config.read(sys.argv[1])

    # virus
    virus_name = config.get("virus", "virus_code")

    # io
    seq_file = config.get("io", "seq_file")
    cls_file = config.get("io", "cls_file")
    outdir = config.get("io", "outdir")

    # seq_rep
    klen = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers") 
    fragmentSize = config.getint("seq_rep", "fragment_size")
    fragmentCount = config.getint("seq_rep", "fragment_count")
    # ........ main evaluation parameters ..............
    fragmentCovs = config.get("seq_rep", "fragment_cov") 
    # ..................................................

    # evaluation
    evalType = config.get("evaluation", "eval_type") # CF or FF only
    testSize = config.getfloat("evaluation", "test_size") 
    cv_folds = config.getint("evaluation", "cv_folds") 
    eval_metric = config.get("evaluation", "eval_metric")
    avrg_metric = config.get("evaluation", "avrg_metric")

    # classifier
    _multi_class = config.get("classifier", "multi_class")
    _tol = config.getfloat("classifier", "tol")
    _lambda = config.getfloat("classifier", "lambda") 
    _l1_ratio = config.getfloat("classifier", "l1_ratio") 
    _solver = config.get("classifier", "solver")
    _max_iter = config.getint("classifier", "max_iter")
    _penalties = config.get("classifier", "penalty")

    # settings 
    nJobs = config.getint("settings", "n_jobs")
    verbose = config.getint("settings", "verbose")
    saveFiles = config.getboolean("settings", "save_files")
    randomState = config.getint("settings", "random_state")

    if evalType not in ["CF", "FF"]:
        raise ValueError(
                "evalType argument have to be one of CF or FF values")

    if fullKmers:
        tag_kf = "F"
    else:
        tag_kf = "S"

    # OutDir folder
    ###############
    outdir = os.path.join(outdir,"{}/{}".format(virus_name, evalType))
    makedirs(outdir, mode=0o700, exist_ok=True)
 
    # Coverage values to evaluate
    #############################
    coverages = str_to_list(fragmentCovs, cast=float)
    coverages_str = [str(c) for c in coverages]

    ## MLR initialization
    ########################################## 

    mlr = LogisticRegression(multi_class=_multi_class, tol=_tol, 
            solver=_solver, max_iter=_max_iter, verbose=0, l1_ratio=None)

    ## Evaluate MLR models
    ######################

    # "l1", "l2", "elasticnet", "none"
    clf_penalties = str_to_list(_penalties)
    clf_names = ["SKMLR_"+pen.upper() for pen in clf_penalties]
    clf_scores = defaultdict(dict) 
    score_names = compile_score_names(eval_metric, avrg_metric)

    parallel = Parallel(n_jobs=len(clf_names), prefer="processes", 
            verbose=verbose)

    # If we have enough memory we can parallelize this loop
    for coverage in coverages:
        if verbose:
            print("\nEvaluating coverage {}".format(coverage), flush=True)

        # Construct prefix for output files
        ###################################

        tag_fg = "FSZ{}_FCV{}_FCL{}_".format(str(fragmentSize), 
                str(coverage), str(fragmentCount))

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
                fragment_size=fragmentSize,
                fragment_cov=coverage,
                fragment_count=fragmentCount,
                n_splits=cv_folds,
                test_size=testSize,
                random_state=randomState,
                verbose=verbose)

        cv_data = tt_data["data"]

        mlr_scores = parallel(delayed(perform_sk_mlr_cv)(clone(mlr), clf_name,
            clf_penalty, _lambda, cv_data, prefix_out, eval_metric,
            avrg_metric, cv_folds, saveFiles, verbose, randomState)
            for clf_name, clf_penalty in zip(clf_names, clf_penalties))
        #print(mlr_scores)

        for i, clf_name in enumerate(clf_names):
            clf_scores[clf_name][str(coverage)] = mlr_scores[i]

    #print(clf_scores)
 
    scores_dfs = make_clf_score_dataframes(clf_scores, coverages_str, 
            score_names, _max_iter)

    #pprint(scores_dfs)

    ## Save and Plot results
    ########################
    tag_cov = "FSZ{}_FCV{}to{}_FCL{}_".format(str(fragmentSize), 
            coverages_str[0], coverages_str[-1], str(fragmentCount))

    outFile = os.path.join(outdir, "{}_{}_K{}{}_{}A{}_MLR_COVERAGES".format(
        virus_name, evalType, tag_kf, klen, tag_cov, _lambda))

    if saveFiles:
        write_log(scores_dfs, config, outFile+".log")

    plot_cv_figure(scores_dfs, score_names, coverages_str, "Coverage",
            outFile)
# print also dataframes into a file
