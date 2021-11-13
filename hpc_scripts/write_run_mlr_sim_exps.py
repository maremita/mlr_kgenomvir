#!/usr/bin/env python

from mlr_kgenomvir.utils import read_config_file

import sys
import os
import os.path
from os import makedirs
from itertools import product
from datetime import datetime
import argparse

__author__ = "amine"

"""
The script create configuration files and run different
experiments to evaluate the effect of different parameters
on the performance and behavior of MLR models
"""


def main(args):

    # Get config values from ini files
    slurm_config = read_config_file(args.slurm_config)
    job_config = read_config_file(args.job_config)
    job_type = args.job_type
    job_name = args.job_name

    account = slurm_config.get("user", "account")
    mail = slurm_config.get("user", "mail_user")

    cpu_task = slurm_config.get("resources", "cpus_per_task")
    gres = slurm_config.get("resources", "gres")
    mem = slurm_config.get("resources", "mem")
    time = slurm_config.get("resources", "time")

    output_folder = slurm_config.get("io", "output_folder")

    # Get or set Job name

    if not job_name:
        now = datetime.now()
        str_time = now.strftime("%m%d")
        job_name = "MLR"+str_time

    print("Runing {} experiments\n".format(job_name))

    #
    k_list = ["4", "9"]
    evaluations = ["CC", "CF", "FF"]
    penalties = ["none", "l2", "l1"]
    frgt_list = ["250", "500"]

    # coverage
    if job_type == "coverage":
        program = "mlr_eval_sim_coverage.py"
        exp_code = "coverage"
        exp_mini = "C"
        #
        exp_section = "seq_rep"
        exp_key = "fragment_cov"
        exp_values = ["0.1", "1", "2", "10"] 

    # mutation
    elif job_type == "mutation":
        program = "mlr_eval_sim_evoparams.py"
        exp_code = "mutation"
        exp_mini = "M"
        #
        exp_section = "simulation"
        exp_key = "mutation_rate"
        exp_values = ["0.0001", "0.001", "0.01", "0.1"] 

    # indel
    elif job_type == "indel":
        program = "mlr_eval_sim_evoparams.py"
        exp_code = "indel"
        exp_mini = "I"
        #
        exp_section = "simulation"
        exp_key = "indel_prob"
        exp_values = ["0.0001", "0.001", "0.01", "0.1"] 

    # recombination
    elif job_type == "recomb":
        program = "mlr_eval_sim_evoparams.py"
        exp_code = "recomb"
        exp_mini = "R"
        job_config.set("simulation", "rep_dual_infection", "0.01")
        #
        exp_section = "simulation"
        exp_key = "rep_recombination"
        exp_values = ["0.0001", "0.001", "0.01", "0.1"] 

    # imbalanced dataset
    elif job_type == "imbdata":
        program = "mlr_eval_sim_imbalanced_dataset.py"
        exp_code = "imbdata"
        exp_mini = "D"
        #
        exp_section = "simulation"
        exp_key = "class_pop_size_std"
        exp_values = ["0", "5", "10", "50"] 

    # imbalanced sampling
    elif job_type == "imbsamp":
        program = "mlr_eval_sim_imbalanced_sampling.py"
        exp_code = "imbsamp"
        exp_mini = "S"
        #
        exp_section = "seq_rep"
        exp_key = "class_size_std"
        exp_values = ["0", "5", "10", "50"] 

    # k lengths 
    elif job_type == "klen":        
        program = "mlr_eval_sim_klen.py"
        exp_code = "klen"
        exp_mini = "K"
        #
        exp_section = "seq_rep"
        exp_key = "k"
        exp_values = ["4", "5", "6", "7", "8", "9", "10"]
        k_list = ["0"] # not important here

    # Lambda
    elif job_type == "lambda":
        program = "mlr_eval_sim_lambda.py"
        exp_code = "lambda"
        exp_mini = "A"
        #
        exp_section = "classifier"
        exp_key = "lambda"
        exp_values = ["1e-3", "1e-1", "1", "1e1", "1e2"] 

    # Learning rate
    elif job_type == "lr":
        program = "mlr_eval_sim_learning_rate.py"
        exp_code = "lr"
        exp_mini = "L"
        #
        exp_section = "classifier"
        exp_key = "learning_rate"
        exp_values = ["1e-5", "1e-3", "1e-1"] 

    # Low variance threshold
    elif job_type == "lowvar":
        program = "mlr_eval_sim_lowVar.py"
        exp_code = "lowvar"
        exp_mini = "V"
        #
        exp_section = "seq_rep"
        exp_key = "low_var_threshold"
        exp_values = ["0.01", "0.1", "0.3", "0.5"] 

    # number of classes
    elif job_type == "nbclass":
        program = "mlr_eval_sim_nbclasses.py"
        exp_code = "nbclass"
        exp_mini = "N"
        #
        exp_section = "simulation"
        exp_key = "nb_classes"
        exp_values = ["2", "5", "10", "20"] 

    else:
        raise ValueError("job_type should be one of these values:\n\n"\
                "[coverage | mutation | indel | recomb | imbdata |\n "\
                "imbsamp | klen | lambda | lr | lowvar | nbclass]\n")


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

    #config_dir = "config_folder/{}/".format(exp_code)
    config_dir = os.path.join(output_folder, "job_confs")
    makedirs(config_dir, mode=0o700, exist_ok=True)

    # Simulated data folder will be created by the main 
    # eval script in "output_folder". So CC, CF and CF 
    # evaluations will use the same simulated data if 
    # load_data flag is set to True in config file, otherwise
    # new data are generated for new executions and previous data
    # will be lost

    #
    for eval_type in evaluations:
        if eval_type in ["CF", "FF"]:
            fgt_sizes = frgt_list
        else:
            fgt_sizes = [0] # not important here

        for frgt_size, k, pen in product(fgt_sizes, k_list, penalties):
            for exp_value in exp_values:
                #
                if exp_code == "klen":
                    k = exp_value
                else:
                    job_config.set('seq_rep', 'k', str(k))
                #
                exp_name = "{}{}_k{}_{}{}_{}".format(
                        exp_mini,
                        exp_value,
                        k,
                        eval_type,
                        frgt_size,
                        pen)

                # Update config parser
                # set the the value of the parameter to evaluate
                job_config.set(exp_section, exp_key, str(exp_value))
                job_config.set('evaluation', 'eval_type', eval_type)
                job_config.set('seq_rep', 'fragment_size', str(frgt_size))
                job_config.set('classifier', 'penalty', pen)

                # write it on a file
                config_file = os.path.join(config_dir, "{}_{}.ini"\
                        "".format(exp_name, job_name))

                with open (config_file, "w") as fh:
                    job_config.write(fh)

                # submit job with this config file
                cmd = "sbatch --account={} --mail-user={} --cpu-per-task={} "\
                        "--job-name={} --time={} --export=PROGRAM={},CONF_file={} "\
                        "--mem={} --gres={} --error={} --output={} submit_mlr_exp.sh".format(
                                account,
                                mail,
                                cpu_task,
                                exp_name, 
                                time,
                                program, config_file, 
                                mem,
                                gres, 
                                s_error,
                                s_output)
                print(cmd, end="\n\n")
                #os.system(cmd)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to write and run evaluation scripts')

    parser.add_argument('-s', '--slurm-config', type=str, required=True)
    parser.add_argument('-c', '--job-config', type=str, required=True)
    parser.add_argument('-t', '--job-type', type=str, required=True)
    parser.add_argument('-n', '--job-name', type=str, required=False)

    args = parser.parse_args()

    main(args)

