#!/usr/bin/env python

import sys
import os
import os.path
from os import makedirs
from itertools import product
from datetime import datetime
import argparse
import configparser


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
    job_name = args.job_name

    account = job_config.get("slurm", "account")
    mail = job_config.get("slurm", "mail_user")

    cpu_task = job_config.get("resources", "cpus_per_task")
    gres = job_config.get("resources", "gres")
    mem = job_config.get("resources", "mem")
    time = job_config.get("resources", "time")
    output_folder = job_config.get("resources", "output_folder")

    # Set job name if not defined
    if not job_name:
        now = datetime.now()
        str_time = now.strftime("%m%d")
        job_name = "MLR"+str_time

    print("Runing {} experiments\n".format(job_name))

    #
    k_list = str_to_list(
            job_config.get("distr_evals", "k_list"), cast=str)
    eval_types = str_to_list(
            job_config.get("distr_evals", "eval_types"), cast=str)
    penalties = str_to_list(
            job_config.get("distr_evals", "penalties"), cast=str)
    fragment_sizes = str_to_list(
            job_config.get("distr_evals", "fragment_sizes"),
            cast=str)
    iter_bounds = str_to_list(
            job_config.get("distr_evals", "sim_iterations"),
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

    # indels
    elif exp_code == "indels":
        program = "mlr_eval_sim_evoparams.py"
        exp_mini = "I"
        #
        exp_section = "simulation"
        exp_key = "indel_prob"

    # recombinations
    elif exp_code == "recombs":
        program = "mlr_eval_sim_evoparams.py"
        exp_mini = "R"
        job_config.set("simulation", "rep_dual_infection",
                job_config.get("distr_evals", "rep_dual_infection"))
        #
        exp_section = "simulation"
        exp_key = "rep_recombination"

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
                "[coverages | mutations | indels | recombs | imbdata |\n "\
                "imbsamp | klens | lambdas | lrs | lowvars | nbclasses]\n")

    exp_values = str_to_list(
            job_config.get("distr_evals", exp_code), cast=str)
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

    s_error = os.path.join(log_dir, "%j.err")
    s_output = os.path.join(log_dir, "%j.out")

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

    #
    for eval_type in eval_types:
        if eval_type in ["CF", "FF"]:
            fgt_sizes = fragment_sizes
        else:
            fgt_sizes = [0] # not important here

        for frgt_size, k, pen, iteration in \
                product(fgt_sizes, k_list, penalties, sim_iterations):
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

                # write it on a file
                config_file = os.path.join(config_dir, "{}_{}.ini"\
                        "".format(exp_name, job_name))

                with open (config_file, "w") as fh:
                    job_config.write(fh)

                # submit job with this config file
                cmd = "sbatch --account={} --mail-user={} --cpus-per-task={} "\
                        "--job-name={} --time={} --export=PROGRAM={},CONF_file={} "\
                        "--mem={} {}--error={} --output={} submit_mlr_exp.sh".format(
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
                print(cmd, end="\n\n")
                #os.system(cmd)


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
    parser.add_argument('-n', '--job-name', type=str,
            required=False)

    args = parser.parse_args()

    main(args)

