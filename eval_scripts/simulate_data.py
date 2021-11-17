#!/usr/bin/env python

from mlr_kgenomvir.data.seq_collections import SeqCollection
from mlr_kgenomvir.data.build_cv_data import build_load_save_cv_data
from mlr_kgenomvir.utils import get_stats
from mlr_kgenomvir.simulation.santasim import SantaSim

import random
import sys
import configparser
import os.path
from os import makedirs
import re

__author__ = ["amine", "nicolas"]


"""
The script simulate and generate un labled dataset using
SantaSim python class (that wraps the amazing tool SANTA-SIM)
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


    # settings
    verbose = config.getint("settings", "verbose",
            fallback=0)
    loadData = config.getboolean("settings", "load_data",
            fallback=False)
    saveData = config.getboolean("settings", "save_data",
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
        # writeSimXMLConfig of SantaSim will check if initSeq 
        # is an integer
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


    # SimDir folder
    ###############
    sim_dir = os.path.join(outdir,"sim_data")
    makedirs(sim_dir, mode=0o700, exist_ok=True)

    # OutDir folder
    ###############
    outdir = os.path.join(outdir,"{}".format(evalType))
    makedirs(outdir, mode=0o700, exist_ok=True)

    # SantaSim object for sequence simulation using SANTA
    sim = SantaSim()

    for iteration in range(1, sim_iter + 1):
        if verbose:
            print("\nEvaluating Simulation {}".format(iteration),
                    flush=True)
        
        # Update random state for each iteration to generate 
        # different datasets
        if isinstance(randomState, (int)):
            randomState *= iteration

        # Construct names for simulation and classes files
        ###################################
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

    if verbose:
        print("\nFin normale du programme {}".format(sys.argv[0]))
