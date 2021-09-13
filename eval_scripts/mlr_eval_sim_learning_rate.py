#!/usr/bin/env python

from mlr_kgenomvir.data.build_cv_data import build_load_save_cv_data
from mlr_kgenomvir.models.model_evaluation import perform_mlr_cv
from mlr_kgenomvir.models.model_evaluation import compile_score_names
from mlr_kgenomvir.models.model_evaluation import make_clf_score_dataframes
from mlr_kgenomvir.models.model_evaluation import plot_cv_figure
from mlr_kgenomvir.utils import str_to_list
from mlr_kgenomvir.utils import write_log
from mlr_kgenomvir.simulation.simulation import SantaSim

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


__author__ = ["amine","nicolas"]


"""
The script evaluates the effect of the learning rate (step size)
on the performance of regularized MLR classifiers for virus genome
classification of a simulated population
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
    outdir = config.get("io", "outdir")

    # seq_rep
    klen = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers")
    lowVarThreshold = config.get("seq_rep", "low_var_threshold", fallback=None)

    # evaluation
    evalType = config.get("evaluation", "eval_type") # CC, CF or FF
    testSize = config.getfloat("evaluation", "test_size")
    cv_folds = config.getint("evaluation", "cv_folds")
    eval_metric = config.get("evaluation", "eval_metric")
    avrg_metric = config.get("evaluation", "avrg_metric")

    # classifier
    _module = config.get("classifier", "module") # sklearn or pytorch_mlr
    _tol = config.getfloat("classifier", "tol")
    _lambda = config.getfloat("classifier", "lambda")
    # ........ main evaluation parameters ..............
    _learning_rates = config.get("classifier", "learning_rate")
    # ..................................................
    _l1_ratio = config.getfloat("classifier", "l1_ratio")
    _solver = config.get("classifier", "solver")
    _max_iter = config.getint("classifier", "max_iter")
    _penalties = config.get("classifier", "penalty")

    if _module == "pytorch_mlr":
        _n_iter_no_change = config.getint("classifier", "n_iter_no_change")
        _device = config.get("classifier", "device")

    # settings
    n_mainJobs = config.getint("settings", "n_main_jobs")
    n_cvJobs = config.getint("settings", "n_cv_jobs")
    verbose = config.getint("settings", "verbose")
    saveData = config.getboolean("settings", "save_data")
    saveModels = config.getboolean("settings", "save_models")
    saveResults = config.getboolean("settings", "save_results")
    plotResults = config.getboolean("settings", "plot_results")
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

    # simulations
    sim_iter = config.getint("simulation", "iterations")
    sim_dir = "{}/simulations".format(outdir)
    sim_config = "{}/simulations_config.xml".format(sim_dir)

    # Check lowVarThreshold
    # #####################
    if lowVarThreshold == "None":
        lowVarThreshold = None
    else:
        lowVarThreshold = float(lowVarThreshold)

    # Construct prefix for output files
    # #################################
    if fullKmers:
        tag_kf = "F"
    elif lowVarThreshold:
        tag_kf = "V"
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


    # SimDir folder
    ###############
    sim_dir = os.path.join(sim_dir,"{}/{}".format(virus_name, evalType))
    makedirs(sim_dir, mode=0o700, exist_ok=True)

    ## Learning rate values to evaluate
    ###################################
    lrs = str_to_list(_learning_rates, cast=float)
    lrs_str = [format(l, '.0e') if l not in list(
        range(0,10)) else str(l) for l in lrs]

    ## MLR initialization
    #####################

    if _module == "pytorch_mlr":
        mlr = MLR(tol=_tol, l1_ratio=None, solver=_solver,
                max_iter=_max_iter, validation=False,
                n_iter_no_change=_n_iter_no_change, device=_device,
                random_state=randomState, verbose=verbose)
        mlr_name = "PTMLR"

    else:
        raise ValueError("module values must be pytorch_mlr."+\
                " Sklearn implementation of MLR does not initialize"+\
                " learning rate")

    ## Evaluate MLR models
    ######################

    # "l1", "l2", "elasticnet", "none"
    clf_penalties = str_to_list(_penalties)
    clf_names = [mlr_name+"_"+pen.upper() for pen in clf_penalties]

    clf_scores = defaultdict(dict)
    score_names = compile_score_names(eval_metric, avrg_metric)

    parallel = Parallel(n_jobs=n_mainJobs, prefer="processes", verbose=verbose)

    # If we have enough memory we can parallelize this loop

    for iteration in range(1,sim_iter + 1):
        if verbose:
            print("\nEvaluating Simulation {}".format(iteration), flush=True)

            # Construct names for simulation and classes files
            ###################################
            sim_name = "simulation_{}".format(iteration)
            cls_file = "{}/class_{}.csv".format(sim_dir, str(iteration))

            # Simulate viral population based on input fasta
            ################################################
            sim = SantaSim(seq_file, cls_file, sim_config, sim_dir, sim_name, virusName = virus_name)
            sim_file = sim.santaSim()

        for i, (clf_name, clf_penalty) in enumerate(zip(clf_names,clf_penalties)):
            if verbose:
                print("\n{}. Evaluating {}".format(i, clf_name), flush=True)

            # Construct prefix for output files
            ###################################
            prefix_out = os.path.join(outdir, "{}_{}_K{}{}_sim{}_{}".format(
            virus_name, evalType, tag_kf, klen, iteration, tag_fg))

            ## Generate training and testing data
            ####################################
            tt_data = build_load_save_cv_data(
                    sim_file,
                    cls_file,
                    prefix_out,
                    eval_type=evalType,
                    k=klen,
                    full_kmers=fullKmers,
                    low_var_threshold=lowVarThreshold,
                    n_splits=cv_folds,
                    test_size=testSize,
                    save_data=saveData,
                    random_state=randomState,
                    verbose=verbose,
                    **args_fg)

            cv_data = tt_data["data"]

            mlr_scores = parallel(delayed(perform_mlr_cv)(clone(mlr), clf_name,
                clf_penalty, _lambda, cv_data, prefix_out, learning_rate=_lr,
                metric=eval_metric, average_metric=avrg_metric, n_jobs=n_cvJobs,
                save_model=saveModels, save_result=saveResults,
                verbose=verbose, random_state=randomState)
                for _lr in lrs)

            for j, lr_str in enumerate(lrs_str):
                clf_scores[clf_name][lr_str] = mlr_scores[j]

        scores_dfs = make_clf_score_dataframes(clf_scores, lrs_str,
                score_names, _max_iter)

        ## Save and Plot results
        ########################
        str_lambda = format(_lambda, '.0e') if _lambda not in list(
                range(0, 10)) else str(_lambda)

        outFile = os.path.join(outdir,
                "{}_{}_K{}{}_{}{}_LR{}to{}_A{}_LRS_sim{}_{}_{}".format(virus_name,
                    evalType, tag_kf, klen, tag_fg, mlr_name, lrs_str[0],
                    lrs_str[-1], str_lambda, iteration, eval_metric, avrg_metric))

        if saveResults:
            write_log(scores_dfs, config, outFile+".log")
            with open(outFile+".jb", 'wb') as fh:
                dump(scores_dfs, fh)

        if plotResults:
            plot_cv_figure(scores_dfs, score_names, lrs_str, "Learning rate",
                    outFile)

    if verbose:
        print("\nFin normale du programme")