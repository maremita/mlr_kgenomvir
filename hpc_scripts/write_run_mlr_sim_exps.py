#!/usr/bin/env python

import sys
import os
import os.path
from os import makedirs
from itertools import product
from datetime import datetime
import argparse
import configparser
from collections import defaultdict

__author__ = "amine"

"""
The script create configuration files and run different
experiments to evaluate the effect of different parameters
on the performance and behavior of MLR models
"""


def main(args):

    # Get config values from ini files
    job_config = read_config_file(args.job_config)
    exp_code = args.job_type
    plot_only = args.plot_only
    submit_job = args.submit

    account = job_config.get("slurm", "account")
    mail = job_config.get("slurm", "mail_user")
    
    job_name = job_config.get("main_settings", "job_name",
            fallback=None)
    cpu_task = job_config.get("main_settings", "cpus_per_task",
            fallback=1)
    gres = job_config.get("main_settings", "gres")
    mem = job_config.get("main_settings", "mem")
    time = job_config.get("main_settings", "time")
    output_folder = job_config.get("main_settings", "output_folder")

    if str(job_name).lower() in ["auto", "none"]:
        job_name = None

    if not job_name:
        now = datetime.now()
        str_time = now.strftime("%m%d")
        job_name = "MLR"+str_time

    #
    k_list = str_to_list(
            job_config.get("main_evals", "k_list"), cast=str)
    eval_types = str_to_list(
            job_config.get("main_evals", "eval_types"), cast=str)
    penalties = str_to_list(
            job_config.get("main_evals", "penalties"), cast=str)
    fragment_sizes = str_to_list(
            job_config.get("main_evals", "fragment_sizes"),
            cast=str)
    iter_bounds = str_to_list(
            job_config.get("main_evals", "sim_iterations"),
            cast=int)

    # coverages
    if exp_code == "coverages":
        program = "mlr_eval_sim_coverage.py"
        exp_mini = "C"
        #
        exp_section = "seq_rep"
        exp_key = "fragment_cov"

    # mutations
    elif exp_code == "mutations":
        program = "mlr_eval_sim_evoparams.py"
        exp_mini = "M"
        #
        exp_section = "simulation"
        exp_key = "mutation_rate"
        #
        job_config.set(exp_section, "evo_to_assess", exp_key)

    # indels
    elif exp_code == "indels":
        program = "mlr_eval_sim_evoparams.py"
        exp_mini = "I"
        #
        exp_section = "simulation"
        exp_key = "indel_prob"
        #
        job_config.set(exp_section, "evo_to_assess", exp_key)

    # recombinations
    elif exp_code == "recombs":
        program = "mlr_eval_sim_evoparams.py"
        exp_mini = "R"
        #
        exp_section = "simulation"
        exp_key = "rep_recombination"
        #
        job_config.set(exp_section, "rep_dual_infection",
                job_config.get("main_evals", "rep_dual_infection", 
                    fallback=0.01))
        job_config.set(exp_section, "evo_to_assess", exp_key)

    # imbalanced dataset
    elif exp_code == "imbdata":
        program = "mlr_eval_sim_imbalanced_dataset.py"
        exp_mini = "D"
        #
        exp_section = "simulation"
        exp_key = "class_pop_size_std"

    # imbalanced sampling
    elif exp_code == "imbsamp":
        program = "mlr_eval_sim_imbalanced_sampling.py"
        exp_mini = "S"
        #
        exp_section = "seq_rep"
        exp_key = "class_size_std"

    # k lengths 
    elif exp_code == "klens":       
        program = "mlr_eval_sim_klen.py"
        exp_mini = "K"
        #
        exp_section = "seq_rep"
        exp_key = "k"
        k_list = ["0"] # not important here

    # Lambdas
    elif exp_code == "lambdas":
        program = "mlr_eval_sim_lambda.py"
        exp_mini = "A"
        #
        exp_section = "classifier"
        exp_key = "lambda"

    # Learning rates
    elif exp_code == "lrs":
        program = "mlr_eval_sim_learning_rate.py"
        exp_mini = "L"
        #
        exp_section = "classifier"
        exp_key = "learning_rate"

    # Low variance thresholds
    elif exp_code == "lowvars":
        program = "mlr_eval_sim_lowVar.py"
        exp_mini = "V"
        #
        exp_section = "seq_rep"
        exp_key = "low_var_threshold"

    # number of classes
    elif exp_code == "nbclasses":
        program = "mlr_eval_sim_nbclasses.py"
        exp_mini = "N"
        #
        exp_section = "simulation"
        exp_key = "nb_classes"

    else:
        raise ValueError(
                "job_type should be one of these values:\n\n"\
                "[coverages | mutations | indels | recombs |"\
                " imbdata |\n imbsamp | klens | lambdas | lrs"\
                " | lowvars | nbclasses]\n")

    exp_dist_evals = job_config.get("main_evals", exp_code)
    exp_values = str_to_list(exp_dist_evals, cast=str)
    job_config.set("job", "job_code", exp_mini)

    # Output folders
    # ###############
 
    # Main output folder
    output_folder = os.path.join(output_folder, exp_code, job_name)
    makedirs(output_folder, mode=0o700, exist_ok=True)
    job_config.set("io", "outdir", output_folder)

    # Job log folder
    log_dir = os.path.join(output_folder, "job_logs")
    makedirs(log_dir, mode=0o700, exist_ok=True)

    # Config files directory
    config_dir = os.path.join(output_folder, "job_confs")
    makedirs(config_dir, mode=0o700, exist_ok=True)

    # Simulated data folder will be created by the main 
    # eval script in "output_folder". So CC, CF and CF 
    # evaluations will use the same simulated data if 
    # load_data flag is set to True in config file, otherwise
    # new data are generated for new executions and previous data
    # will be lost

    # Set sim iteration bounderies (useful when simulated
    # data are deterministic) 
    if len(iter_bounds) == 2:
        # Config ex: iterations = 1, 10
        sim_iter_start = iter_bounds[0]
        sim_iter_end = iter_bounds[1] + 1

    elif len(iter_bounds) == 1:
        # Config ex: iterations = 5
        sim_iter_start = 1
        sim_iter_end = iter_bounds[0] + 1

    sim_iterations = list(range(sim_iter_start, sim_iter_end))

    # gres (Generic RESources)
    # Options: cpu or gpu:1 (beluga) or gpu:v100l:1 (cedar) etc.
    set_gres = ""
    if "gpu" in gres:
        set_gres = "--gres={} ".format(gres)
        job_config.set('classifier', 'device', "cuda")

    else:
        job_config.set('classifier', 'device', "cpu")

    # Set parameter to run computations or plot results already done
    job_config.set('settings', 'plot_results_only', str(plot_only))

    # Run computations
    if not plot_only:
        print("Running {} {} experiments\n".format(job_name, exp_code))

        nb_runs = defaultdict(lambda: 0)
        for eval_type in eval_types:
            if eval_type in ["CF", "FF"]:
                fgt_sizes = fragment_sizes
            else:
                fgt_sizes = [0] # not important here

            for frgt_size, k, pen, iteration in \
                    product(fgt_sizes, k_list, penalties,
                            sim_iterations):

                kef = "{}_{}_{}".format(k, eval_type, frgt_size)
                if kef not in nb_runs:
                    nb_runs[kef] = 0

                for exp_value in exp_values:
                    #
                    if exp_code == "klens":
                        k = exp_value
                    else:
                        job_config.set('seq_rep', 'k', str(k))
                    #
                    exp_name = "S{}{}{}_K{}{}{}_{}".format(
                            iteration,
                            exp_mini,
                            exp_value,
                            k,
                            eval_type,
                            frgt_size,
                            pen)

                    # Update config parser
                    # set the the value of the parameter to evaluate
                    job_config.set(exp_section, exp_key, str(exp_value))
                    job_config.set('simulation', 'iterations',
                            "{},{}".format(iteration, iteration))
                    job_config.set('evaluation', 'eval_type', eval_type)
                    job_config.set('seq_rep', 
                            'fragment_size', str(frgt_size))
                    job_config.set('classifier', 'penalty', pen)

                    measure_files, prefix_mfile = get_measure_file_names(
                            job_config,
                            exp_section,
                            exp_key,
                            iteration)

                    dont_run = all(
                            [os.path.isfile(f) for f in measure_files])

                    if not dont_run:
                        # write config on a file
                        config_file = os.path.join(
                                config_dir, "{}_{}.ini".format(
                                    os.path.basename(prefix_mfile), job_name))

                        with open (config_file, "w") as fh:
                            job_config.write(fh)

                        s_error = os.path.join(log_dir,
                                "%j_" + os.path.basename(prefix_mfile) + ".err")
                        s_output = os.path.join(log_dir,
                                "%j_" + os.path.basename(prefix_mfile) + ".out")

                        # submit job with this config file
                        cmd = "sbatch --account={} --mail-user={} "\
                                "--cpus-per-task={} --job-name={} "\
                                "--time={} "\
                                "--export=PROGRAM={},CONF_file={} "\
                                "--mem={} {}--error={} --output={} "\
                                "submit_mlr_exp.sh".format(
                                        account,
                                        mail,
                                        cpu_task,
                                        exp_name, 
                                        time,
                                        program, config_file, 
                                        mem,
                                        set_gres, 
                                        s_error,
                                        s_output)

                        print(prefix_mfile)

                        if submit_job:
                            print(cmd, end="\n")
                            os.system(cmd)
                            print("\n")
                            nb_runs[kef] += 1

        print("Number of runs:")
        total = 0
        for run_type in nb_runs:
            print("{}: {}".format(run_type, nb_runs[run_type]))
            total += nb_runs[run_type]
        print("Total: {}".format(total))

    ## If all computations are done
    ## Aggregate results and generate plots
    else:
        print("Plotting {} {} experiments\n".format(job_name, exp_code))

        for eval_type in eval_types:
            if eval_type in ["CF", "FF"]:
                fgt_sizes = fragment_sizes
            else:
                fgt_sizes = [0] # not important here

            for frgt_size, k in product(fgt_sizes, k_list):

                kef = "{}_{}_{}".format(k, eval_type, frgt_size)
                #
                if exp_code != "klens":
                    job_config.set('seq_rep', 'k', str(k))
                #
                exp_name = "{}_K{}_{}{}".format(
                        exp_mini,
                        k,
                        eval_type,
                        frgt_size)

                # Update config parser
                # set the the values of the parameter to evaluate
                job_config.set(exp_section, exp_key, exp_dist_evals)
                job_config.set('simulation',
                        'iterations', str(sim_iter_end-1))
                job_config.set('evaluation',
                        'eval_type', eval_type)
                job_config.set('seq_rep',
                        'fragment_size', str(frgt_size))
                job_config.set('classifier', 
                        'penalty', ", ".join(penalties))
 
                # write config on a file
                config_file = os.path.join(
                        config_dir, "{}_{}.ini".format(
                            exp_name, job_name))
                
                print(config_file)

                with open (config_file, "w") as fh:
                    job_config.write(fh)

                s_error = os.path.join(log_dir,
                        "%j_" + exp_name + ".err")
                s_output = os.path.join(log_dir,
                        "%j_" + exp_name + ".out")

                # submit job with this config file
                cmd = "sbatch --account={} --mail-user={} "\
                        "--cpus-per-task={} --job-name={} "\
                        "--time={} "\
                        "--export=PROGRAM={},CONF_file={} "\
                        "--mem={} {}--error={} --output={} "\
                        "submit_mlr_exp.sh".format(
                                account,
                                mail,
                                cpu_task,
                                exp_name, 
                                time,
                                program, config_file, 
                                mem,
                                set_gres, 
                                s_error,
                                s_output)


                if submit_job:
                    print(cmd, end="\n")
                    os.system(cmd)

def get_measure_file_names(
        config,
        exp_section,
        exp_key,
        sim_iteration):

    # Get some config parameters
    outdir = config.get("io", "outdir")
    job_code = config.get("job", "job_code")
    klen = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers")
    lowVarThreshold = config.get("seq_rep", "low_var_threshold")
    fragmentSize = config.getint("seq_rep", "fragment_size")
    fragmentCount = config.getint("seq_rep", "fragment_count")
    fragmentCov = config.getfloat("seq_rep", "fragment_cov")
    evalType = config.get("evaluation", "eval_type")
    cv_folds = config.getint("evaluation", "cv_folds")

    _module = config.get("classifier", "module")
    _lambda = config.getfloat("classifier", "lambda")
    _pen = config.get("classifier", "penalty")

    if _module == "pytorch_mlr":
        _lr = config.getfloat("classifier",
                "learning_rate")

    outdir = os.path.join(outdir,"{}".format(evalType))

    # Generate Sim name
    # #################
    sim_name = "Sim{}".format(sim_iteration)

    sim_value = config.getfloat(exp_section, exp_key)

    if config.has_option(exp_section, "evo_to_assess"):
        # TODO Support str conversion of indelModelNB 
        sim_value = str(sim_value) if sim_value in list(range(0, 10))\
                else format(sim_value, '.0e') 
        sim_name += "_EV{}".format(sim_value)

    elif exp_key == "class_pop_size_std":
        sim_name += "_PSTD{}".format(sim_value)

    elif exp_key == "nb_classes":
        sim_name += "_CL{}".format(sim_value)

    # Check lowVarThreshold
    # #####################
    if lowVarThreshold == "None":
        lowVarThreshold = None
    else:
        lowVarThreshold = float(lowVarThreshold)

    # Generate prefix of output files
    # ###############################
    if fullKmers:
        tag_kf = "F"
    elif lowVarThreshold:
        tag_kf = "V"
    else:
        tag_kf = "S"

    tag_fg = ""
    if evalType in ["CF", "FF"]:
        tag_fg = "FSZ{}_FCV{}_FCL{}_".format(
                str(fragmentSize),
                str(fragmentCov),
                str(fragmentCount))

    prefix_out = os.path.join(
            outdir,
            "{}_{}_{}_K{}{}_{}cv{}_".format(
                job_code,
                evalType,
                sim_name, 
                tag_kf,
                klen,
                tag_fg, 
                cv_folds))

    # Other appends prefix_out
    if exp_key == "low_var_threshold":
        prefix_out += "V{}_".format(
                config.get(exp_section, exp_key))

    elif exp_key == "class_size_std": 
        prefix_out += "CSTD{}_".format(
                config.get(exp_section, exp_key))

    # Generate model name
    # ###################
    if _module == "pytorch_mlr":
        mlr_name = "PTMLR"

    else:
        mlr_name = "SKMLR"

    clf_name = mlr_name+"_"+_pen.upper()

    if _module == "pytorch_mlr":
        str_lr = format(_lr, '.0e') if _lr not in list(
                range(0, 10)) else str(_lr)
        clf_name += "_LR"+str_lr

    if _pen != "none":
        str_lambda = format(_lambda, '.0e') if _lambda not in list(
                range(0, 10)) else str(_lambda)
        clf_name += "_A"+str_lambda

    prename = prefix_out+clf_name
    measure_files = [prename+"_fold{}".format(fold)+\
            ".npz" for fold in range(cv_folds)]

    return measure_files, prename

def read_config_file(conf_file):

    cfg = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(conf_file, "r") as cf:
        cfg.read_file(cf)

    return cfg

def str_to_list(chaine, sep=",", cast=None):
    c = lambda x: x
    if cast: c = cast

    return [c(i.strip()) for i in chaine.split(sep)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Script to write and run evaluation scripts')

    parser.add_argument('-c', '--job-config', type=str,
            required=True)
    parser.add_argument('-t', '--job-type', type=str,
            required=True)
    parser.add_argument('--plot-only', dest='plot_only',
            action='store_true')
    parser.add_argument('--submit', dest='submit',
            action='store_true')

    args = parser.parse_args()

    main(args)

    print("\nFin normale du programme")
