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
from mlr_kgenomvir.simulation.santasim import SantaSim

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
    ###################################
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(config_file, "r") as cf:
        config.read_file(cf)

    # job code
    job_code = config.get("job", "job_code")

    # io
    seq_file = config.get("io", "seq_file", fallback=None)
    outdir = config.get("io", "outdir")

    # seq_rep
    klen = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers")
    lowVarThreshold = config.get("seq_rep", "low_var_threshold",
            fallback=None)
    fragmentSize = config.getint("seq_rep", "fragment_size",
            fallback=1000)
    fragmentCount = config.getint("seq_rep", "fragment_count",
            fallback=1000)
    fragmentCov = config.getfloat("seq_rep", "fragment_cov",
            fallback=2)
    # Paramters for sampling dataset
    class_size_min = config.getint("seq_rep", "class_size_min",
            fallback=5)
    class_size_max = config.getint("seq_rep", "class_size_max",
            fallback=200)
    class_size_mean = config.getint("seq_rep", "class_size_mean",
            fallback=50)
    class_size_std = config.getfloat("seq_rep", "class_size_std",
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
    # ........ main evaluation parameters ..............
    _learning_rates = config.get("classifier", "learning_rate")
    # ..................................................
    _l1_ratio = config.getfloat("classifier", "l1_ratio")
    _solver = config.get("classifier", "solver")
    _max_iter = config.getint("classifier", "max_iter")
    _penalties = config.get("classifier", "penalty")

    if _module == "pytorch_mlr":
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

    # simulations
    sim_iter = config.getint("simulation", "iterations")
    init_seq = config.get("simulation", "init_seq") # file/none
    init_seq_size = config.getint("simulation", "init_seq_size",
            fallback=None)
    init_gen_count_fraction = config.getfloat("simulation",
            "init_gen_count_fraction", fallback=0.5)

    nb_classes = config.getint("simulation", 
            "nb_classes", fallback=5)
    class_pop_size = config.getint("simulation", 
            "class_pop_size", fallback=25)
    class_pop_size_min = config.getint("simulation",
            "class_pop_size_min", fallback=50)
    class_pop_size_max = config.getint("simulation",
            "class_pop_size_max", fallback=100)
    class_pop_size_std = config.getfloat("simulation",
            "class_pop_size_std", fallback=None)

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
        # writeSimXMLConfig of SantaSim will check if initSeq is 
        # an integer
        initseq = init_seq_size

    # Sampling original dataset
    ###########################
    sampling_args = dict()
    sample_classes = False

    if bool(class_size_std):
        sample_classes = True

    sampling_args = {
            'sample_classes':sample_classes,
            'sample_class_size_min':class_size_min,
            'sample_class_size_max':class_size_max,
            'sample_class_size_mean':class_size_mean,
            'sample_class_size_std':class_size_std
            }

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

    str_lambda = format(_lambda, '.0e') if _lambda not in list(
            range(0, 10)) else str(_lambda)

    # SimDir folder
    ###############
    sim_dir = os.path.join(outdir,"sim_data")
    makedirs(sim_dir, mode=0o700, exist_ok=True)

    # OutDir folder
    ###############
    outdir = os.path.join(outdir,"{}".format(evalType))
    makedirs(outdir, mode=0o700, exist_ok=True)

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
        raise ValueError(
                "module values must be pytorch_mlr."+\
                " Sklearn implementation of MLR does"+\
                " not initialize learning rate")

    ## Evaluate MLR models
    ######################
    # "l1", "l2", "elasticnet", "none"
    clf_penalties = str_to_list(_penalties)
    clf_names = [mlr_name+"_"+pen.upper() for pen in clf_penalties]

    clf_scores = defaultdict(dict)
    score_names = compile_score_names(eval_metric, avrg_metric)

    parallel = Parallel(n_jobs=n_mainJobs, prefer="processes",
            verbose=verbose)

    # SantaSim object for sequence simulation using SANTA
    sim = SantaSim()

    # Collect score results of all simulation iterations in
    # sim_scores
    sim_scores = []

    for iteration in range(1, sim_iter+1):
        if verbose:
            print("\nEvaluating Simulation {}".format(iteration),
                    flush=True)

        # Construct names for simulation and classes files
        ##################################################
        sim_name = "Sim{}".format(iteration)

        # Simulate viral population based on input fasta
        ################################################
        sim_file, cls_file = sim.sim_labeled_dataset(
                [initseq],
                evo_params,
                sim_dir,
                sim_name,
                init_gen_count_frac=init_gen_count_fraction,
                nb_classes=nb_classes,
                class_pop_size=class_pop_size,
                class_pop_size_std=class_pop_size_std,
                class_pop_size_min=class_pop_size_min,
                class_pop_size_max=class_pop_size_max,
                load_data=loadData,
                random_state=randomState,
                verbose=verbose)

        # Construct prefix for output files
        ###################################
        prefix_out = os.path.join(outdir,
                "{}_{}_{}_K{}{}_{}".format(job_code,
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
                **sampling_args,
                **args_fg)

        cv_data = tt_data["data"]
            
        if verbose:
            print("X_train descriptive stats:\n{}".format(
                get_stats(cv_data["X_train"])))

        for i, (clf_name, clf_penalty) in enumerate(
                zip(clf_names,clf_penalties)):
            if verbose:
                print("\n{}. Evaluating {}".format(i+1, clf_name),
                        flush=True)

            ## Train and compute performance of classifiers
            ###############################################
            mlr_scores = parallel(delayed(perform_mlr_cv)(
                clone(mlr), clf_name, clf_penalty, _lambda, cv_data,
                prefix_out, learning_rate=_lr, metric=eval_metric,
                average_metric=avrg_metric, n_jobs=n_cvJobs, 
                load_model=loadModels, save_model=saveModels,
                load_result=loadResults, save_result=saveResults,
                verbose=verbose, random_state=randomState) 
                for _lr in lrs)

            # Add the scores of current clf_name to clf_scores
            for j, lr_str in enumerate(lrs_str):
                clf_scores[clf_name][lr_str] = mlr_scores[j]

        # Rearrange clf_scores into dict of mean and std dataframes
        scores_dfs = make_clf_score_dataframes(clf_scores, 
                lrs_str, score_names, _max_iter)

        sim_scores.append(scores_dfs)

        ## Save and Plot iteration results
        ##################################
        outFileSim = os.path.join(outdir,
                "{}_{}_{}_K{}{}_{}{}_LR{}to{}_A{}_LRS_{}_{}".\
                        format(job_code, evalType, sim_name,
                            tag_kf, klen, tag_fg, mlr_name,
                            lrs_str[0], lrs_str[-1], str_lambda,
                            avrg_metric, eval_metric))

        if saveResults:
            write_log(scores_dfs, config, outFileSim+".log")
            with open(outFileSim+".jb", 'wb') as fh:
                dump(scores_dfs, fh)

        if plotResults:
            plot_cv_figure(scores_dfs, score_names, lrs_str, 
                    "Learning rate", outFileSim)
        
    # Compute the mean of all scores per classifier
    # and by mean and std
    sim_scores_dfs = average_scores_dataframes(sim_scores)

    ## Save and Plot results
    ########################
    outFile = os.path.join(outdir,
            "{}_{}_Sim_K{}{}_{}{}_LR{}to{}_A{}_LRS_{}_{}".format(
                job_code, evalType, tag_kf, klen, tag_fg,
                mlr_name, lrs_str[0], lrs_str[-1], str_lambda,
                avrg_metric, eval_metric))

    if saveResults:
        write_log(sim_scores_dfs, config, outFile+".log")
        with open(outFile+".jb", 'wb') as fh:
            dump(sim_scores_dfs, fh)

    if plotResults:
        plot_cv_figure(sim_scores_dfs, score_names, lrs_str, 
                "Learning rate", outFile)

    if verbose:
        print("\nFin normale du programme {}".format(sys.argv[0]))
