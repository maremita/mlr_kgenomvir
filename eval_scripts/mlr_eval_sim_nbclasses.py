#!/usr/bin/env python

from mlr_kgenomvir.data.seq_collections import SeqCollection
from mlr_kgenomvir.data.build_cv_data import build_load_save_cv_data
from mlr_kgenomvir.models.model_evaluation import perform_mlr_cv
from mlr_kgenomvir.models.model_evaluation import compile_score_names
from mlr_kgenomvir.models.model_evaluation import make_clf_score_dataframes
from mlr_kgenomvir.models.model_evaluation import average_scores_dataframes
from mlr_kgenomvir.models.model_evaluation import plot_cv_figure
from mlr_kgenomvir.utils import str_to_list
from mlr_kgenomvir.utils import get_stats
from mlr_kgenomvir.utils import write_log
from mlr_kgenomvir.simulation.simulation import SantaSim

import random
import sys
import configparser
import os.path
from os import makedirs
from collections import defaultdict
from pprint import pprint
import re

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, dump
from sklearn.base import clone

# Models to evaluate
from mlr_kgenomvir.models.pytorch_mlr import MLR
from sklearn.linear_model import LogisticRegression


__author__ = ["amine"]


"""
The script evaluates the performance of different 
regularized MLR models with function to the number
of classes for virus genome classification of a
simulated population
"""


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Config file is missing!!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    # Get argument values from ini file
    ###################################
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(config_file, "r") as cf:
        config.read_file(cf)

    # virus
    virus_name = config.get("virus", "virus_code")

    # io
    seq_file = config.get("io", "seq_file", fallback=None)
    outdir = config.get("io", "outdir")

    # seq_rep
    klen = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers")
    lowVarThreshold = config.get("seq_rep", "low_var_threshold",
            fallback=None)

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

    if evalType in ["CC", "CF", "FF"]:
        if evalType in ["CF", "FF"]:
            try:
                fragmentSize = config.getint("seq_rep",
                        "fragment_size")
                fragmentCount = config.getint("seq_rep", 
                        "fragment_count")
                fragmentCov = config.getfloat("seq_rep",
                        "fragment_cov")

            except configparser.NoOptionError:
                raise configparser.NoOptionError()
    else:
        raise ValueError(
                "evalType argument have to be one of CC, CF or"+
                " FF values")

    # simulations
    sim_iter = config.getint("simulation", "iterations")
    init_seq = config.get("simulation", "init_seq") # file/none
    init_seq_size = config.getint("simulation", "init_seq_size",
            fallback=None)
    init_gen_count_fraction = config.getfloat("simulation",
            "init_gen_count_fraction", fallback=0.5)
    # ........ main evaluation parameters ..............
    nb_class_list = config.get("simulation", "nb_classes")
    # ..................................................
    class_pop_size = config.getint("simulation", "class_pop_size")
    evo_params = dict()
    evo_params["populationSize"] = config.getint("simulation", 
            "init_pop_size", fallback=100)
    evo_params["generationCount"] = config.getint("simulation", 
            "generation_count", fallback=100)
    evo_params["fitnessFreq"] = config.getfloat("simulation",
            "fitness_freq", fallback=0.5)
    evo_params["repDualInfection"] = config.getfloat("simulation",
            "rep_dual_infection", fallback=0.0)
    evo_params["repRecombination"] = config.getfloat("simulation", 
            "rep_recombination", fallback=0.0)
    evo_params["mutationRate"] = config.getfloat("simulation", 
            "mutation_rate", fallback=0.5)
    evo_params["transitionBias"] = config.getfloat("simulation", 
            "transition_bias", fallback=5.0)
    evo_params["indelModelNB"] = config.get("simulation", 
            "indel_model_nb", fallback=None)
    evo_params["indelProb"] = config.getfloat("simulation", 
            "indel_prob", fallback=None)
 
    # TODO Validate all evo params elsewhere
    # Validate indelModelNB
    if evo_params["indelModelNB"] is not None:
        nb_params = re.split(r'\s+', evo_params["indelModelNB"])
        if len(nb_params) != 2:
            print("\nWarning: indel_model_nb parameter is "\
                    "not valid and set to None\n")
            evo_params["indelModelNB"] = None
            evo_params["indelProb"] = None

    # Here we set the initial seq for all iterations.
    # Maybe we need to use different init seq for each iteration
    # simulation init
    if init_seq == "file":
        simseqs = SeqCollection.read_bio_file(seq_file)
        initseq = simseqs[random.randint(0, len(simseqs))]
    else:
        # writeSimXMLConfig of SantaSim will check if initSeq
        # is an integer
        initseq = init_seq_size

    # Check lowVarThreshold
    # #####################
    if lowVarThreshold == "None":
        lowVarThreshold = None
    else:
        lowVarThreshold = float(lowVarThreshold)

    ## Tags for prefix outputs
    ##########################
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

    str_lr = ""
    if _module == "pytorch_mlr":
        str_lr = format(_learning_rate, '.0e')\
                if _learning_rate not in list(range(0, 10))\
                else str(_learning_rate)
        str_lr = "_LR"+str_lr

    str_lambda = format(_lambda, '.0e') if _lambda not in list(
            range(0, 10)) else str(_lambda)

    # OutDir folder
    ###############
    outdir = os.path.join(outdir,"{}/{}".format(virus_name, 
        evalType))
    makedirs(outdir, mode=0o700, exist_ok=True)

    # SimDir folder
    ###############
    sim_dir = os.path.join(outdir,"simulations")
    makedirs(sim_dir, mode=0o700, exist_ok=True)

    ## Class nbs to evaluate
    ########################
    class_nbs = str_to_list(nb_class_list, cast=int)
    class_nbs_str = [str(c) for c in class_nbs]

    ## MLR initialization
    #####################
    if _module == "pytorch_mlr":
        mlr = MLR(tol=_tol, learning_rate=_learning_rate,
                l1_ratio=None, solver=_solver, max_iter=_max_iter,
                validation=False,
                n_iter_no_change=_n_iter_no_change, device=_device,
                random_state=randomState, verbose=verbose)
        mlr_name = "PTMLR"

    else:
        mlr = LogisticRegression(multi_class="multinomial", 
                tol=_tol, solver=_solver, max_iter=_max_iter,
                verbose=0, l1_ratio=None)
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

    # Collect score results of all simulation iterations in
    # sim_scores
    sim_scores = []

    for iteration in range(1, sim_iter+1):
        if verbose:
            print("\nEvaluating Simulation {}".format(iteration),
                    flush=True)

        for class_nb in class_nbs:
            if verbose:
                print("\nEvaluating class number {}".format(
                    class_nb), flush=True)

            # Construct names for simulation and classes files
            ##################################################
            sim_name = "Sim{}_CL{}".format(iteration, class_nb)

            # Simulate viral population based on input fasta
            ################################################
            sim = SantaSim([initseq], init_gen_count_fraction,
                    nb_classes, class_pop_size, evo_params, sim_dir,
                    sim_name, load_data=loadData, verbose=verbose)
            sim_file, cls_file = sim()

            # Construct prefix for output files
            ###################################
            prefix_out = os.path.join(outdir, 
                    "{}_{}_{}_K{}{}_{}".format(virus_name,
                        evalType, sim_name, tag_kf, klen, tag_fg))

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
                    load_data=loadData,
                    save_data=saveData,
                    random_state=randomState,
                    verbose=verbose,
                    **args_fg)

            cv_data = tt_data["data"]

            if verbose:
                print("X_train descriptive stats:\n{}".format(
                    get_stats(cv_data["X_train"])))
            
            ## Train and compute performance of classifiers
            ###############################################
            mlr_scores = parallel(delayed(perform_mlr_cv)(
                clone(mlr), clf_name, clf_penalty, _lambda,
                cv_data, prefix_out, metric=eval_metric,
                average_metric=avrg_metric, n_jobs=n_cvJobs,
                load_model=loadModels, save_model=saveModels,
                load_result=loadResults, save_result=saveResults,
                verbose=verbose, random_state=randomState)
                for clf_name, clf_penalty in zip(clf_names,
                    clf_penalties))

            # Add the scores of current class_nb to clf_scores
            for i, clf_name in enumerate(clf_names):
                clf_scores[clf_name][str(class_nb)] = mlr_scores[i]

        # Rearrange clf_scores into dict of mean and std dataframes
        scores_dfs = make_clf_score_dataframes(clf_scores,
                class_nbs_str, score_names, _max_iter)
 
        sim_scores.append(scores_dfs)

        ## Save and Plot iteration results
        ##################################
        outFileSim = os.path.join(outdir,
                "{}_{}_Sim{}_CL{}to{}_K{}{}_{}{}{}_A{}_CLASSNBS_{}_{}".format(
                    virus_name, evalType, iteration, class_nbs[0],
                    class_nbs[-1], tag_kf, klen, tag_fg, mlr_name,
                    str_lr, str_lambda, avrg_metric, eval_metric))

        if saveResults:
            write_log(scores_dfs, config, outFileSim+".log")
            with open(outFileSim+".jb", 'wb') as fh:
                dump(scores_dfs, fh)

        if plotResults:
            plot_cv_figure(scores_dfs, score_names, class_nbs_str,
                    "Number of classes", outFileSim)

    # Compute the mean of all scores per classifier
    # and by mean and std
    sim_scores_dfs = average_scores_dataframes(sim_scores)

    ## Save and Plot final results
    ##############################
    outFile = os.path.join(outdir,
                "{}_{}_Sim_CL{}to{}_K{}{}_{}{}{}_A{}_CLASSNBS_{}_{}".\
                        format(virus_name, evalType, class_nbs[0], 
                            class_nbs[-1], tag_kf, klen, tag_fg,
                            mlr_name, str_lr, str_lambda,
                            avrg_metric, eval_metric))

    if saveResults:
        write_log(sim_scores_dfs, config, outFile+".log")
        with open(outFile+".jb", 'wb') as fh:
            dump(sim_scores_dfs, fh)

    if plotResults:
        plot_cv_figure(sim_scores_dfs, score_names, class_nbs_str,
                "Number of classes", outFile)

    if verbose:
        print("\nFin normale du programme {}".format(sys.argv[0]))
