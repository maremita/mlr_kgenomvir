#!/usr/bin/env python

from mlr_kgenomvir.models.model_evaluation import get_mlr_cv_from_files
from mlr_kgenomvir.models.model_evaluation import compile_score_names
from mlr_kgenomvir.models.model_evaluation import make_clf_score_dataframes
from mlr_kgenomvir.models.model_evaluation import plot_cv_figure
from mlr_kgenomvir.utils import str_to_list

import sys
import configparser
import os.path
from os import makedirs
from collections import defaultdict

from joblib import dump, load

__author__ = "amine"


"""
The script compile results data generated by 
eval_scripts/mlr_eval_cv_lambda.py and plot these results.
It's input is the same config file used for 
eval_scripts/mlr_eval_cv_lambda.py for a given experiment
"""


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Config file is missing!!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    # TODO Read config string from STDIN
    # Get argument values from ini file
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
 
    with open(config_file, "r") as cf:
        config.read_file(cf)

    # virus
    virus_name = config.get("virus", "virus_code")

    # io
    # Mandatory directory for models and measures files
    outdir = config.get("io", "outdir")
    # Optional directory for final results and plots
    results_dir = config.get("io", "results_dir", fallback=outdir)

    # seq_rep
    # ........ main evaluation parameters ..............
    #k_lenghts = config.get("seq_rep", "k")
    klen = config.getint("seq_rep", "k")
    
    # ..................................................
    fullKmers = config.getboolean("seq_rep", "full_kmers") 
    lowVarThreshold = config.get("seq_rep", "low_var_threshold", fallback=None)

    # evaluation
    evalType = config.get("evaluation", "eval_type") # CC, CF or FF
    cv_folds = config.getint("evaluation", "cv_folds") 
    eval_metric = config.get("evaluation", "eval_metric")
    avrg_metric = config.get("evaluation", "avrg_metric")

    # classifier
    _module = config.get("classifier", "module") # sklearn or pytorch_mlr
    #_lambda = config.getfloat("classifier", "lambda") 
    _lambdas = config.get("classifier", "lambda")
    
    _max_iter = config.getint("classifier", "max_iter")
    _penalties = config.get("classifier", "penalty")
    _learning_rate = config.getfloat("classifier", "learning_rate", fallback=None)

    # settings 
    verbose = config.getint("settings", "verbose")

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

    # Check lowVarThreshold
    # #####################
    if lowVarThreshold == "None":
        lowVarThreshold = None
    else:
        lowVarThreshold = float(lowVarThreshold)

    ## Tags for prefix out
    ######################
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
    # outdir sould exist
    if not os.path.isdir(outdir):
        raise OSError("{} does not found".format(outdir))

    makedirs(results_dir, mode=0o700, exist_ok=True)

    ## K lengths to evaluate
    ########################
    #klen_list = str_to_list(k_lenghts, cast=int)
    #klen_list_str = [str(k) for k in klen_list]

    ## Lambda values to evaluate
    ############################
    lambdas = str_to_list(_lambdas, cast=float)
    lambdas_str = [format(l, '.0e') if l not in list(
        range(0,10)) else str(l) for l in lambdas]

    ## Output file name
    ###################
    if _module == "pytorch_mlr":
        mlr_name = "PTMLR"
    else:
        mlr_name = "SKMLR"

    str_lr = ""
    if _module == "pytorch_mlr":
        str_lr = format(_learning_rate, '.0e') if _learning_rate not in list(
                range(0, 10)) else str(_learning_rate)
        str_lr = "_LR"+str_lr

    #str_lambda = format(_lambda, '.0e') if _lambda not in list(
    #        range(0, 10)) else str(_lambda)

    outFile = os.path.join(results_dir,
            "{}_{}_K{}{}_{}{}{}_A{}to{}_LAMBDAS_{}_{}".format(virus_name,
                evalType, tag_kf, klen, tag_fg, mlr_name, str_lr,
                lambdas_str[0], lambdas_str[-1], eval_metric, avrg_metric))
    
    #outFile = os.path.join(results_dir, 
    #        "{}_{}_K{}{}to{}_{}{}{}_A{}_KLENGTHS_{}_{}".format(virus_name,
    #            evalType, tag_kf, klen_list[0], klen_list[-1], tag_fg,
    #            mlr_name, str_lr, str_lambda, eval_metric, avrg_metric))

    score_names = compile_score_names(eval_metric, avrg_metric)
    scores_file = outFile+".jb" 

    if os.path.isfile(scores_file):
        if verbose:
            print("Loading score file {}".format(scores_file))
        with open(scores_file, 'rb') as fh:
            scores_dfs = load(fh)

    else:
        # "l1", "l2", "elasticnet"
        clf_penalties = str_to_list(_penalties)
        clf_names = [mlr_name+"_"+pen.upper() for pen in clf_penalties]
        clf_scores = defaultdict(dict)

        prefix_out = os.path.join(outdir, "{}_{}_K{}{}_{}".format(
            virus_name, evalType, tag_kf, klen, tag_fg))

        for l, _lambda in enumerate(lambdas):

            mlr_scores = []

            for clf_name, clf_penalty in zip(clf_names, clf_penalties):
                avg_scores = get_mlr_cv_from_files(clf_name, clf_penalty, _lambda,
                        prefix_out, cv_folds, learning_rate=_learning_rate,
                        metric=eval_metric, average_metric=avrg_metric, 
                        verbose=verbose)
                mlr_scores.append(avg_scores)

            for i, clf_name in enumerate(clf_names):
                clf_scores[clf_name][lambdas_str[l]] = mlr_scores[i]
 
        scores_dfs = make_clf_score_dataframes(clf_scores, lambdas_str, 
                score_names, _max_iter)

        # Save results
        ##############
        if verbose:
            print("Saving scores file {}".format(scores_file))

        with open(scores_file, 'wb') as fh:
            dump(scores_dfs, fh)

    if verbose:
        print("Plotting {}".format(outFile))

    plot_cv_figure(scores_dfs, score_names, lambdas_str, "Lambda", 
            outFile)

    if verbose:
        print("\nFin normale du programme")
