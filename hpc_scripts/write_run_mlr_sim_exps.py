#!/usr/bin/env python

from mlr_kgenomvir.utils import read_config_file

import sys
import os
import os.path
from os import makedirs
from itertools import product
from datetime import datetime


__author__ = "amine"

"""
The script create configuration files and run different
experiments to evaluate the effect of different parameters
on the performance and behavior of MLR models
"""


def main(args):

    # Get config values from ini files
    slurm_config = read_config_file(args[1])
    job_config = read_config_file(args[2])
    exp_type = args[3]

    account = slurm_config.get("user", "account")
    mail = slurm_config.get("user", "mail_user")

    cpu_task = slurm_config.get("resources", "cpus_per_task")
    gres = slurm_config.get("resources", "gres")
    mem = slurm_config.get("resources", "mem")
    time = slurm_config.get("resources", "time")

    output_folder = slurm_config.get("io", "output_folder")

    # Get or set Job name
    job_name = args[4]

    if job_name:
        str_time = job_name
        scores_from_file = "True"
    else:
        now = datetime.now()
        str_time = now.strftime("%m%d%H%M")
        scores_from_file = "False"

    print("Runing {} experiments\n".format(str_time))

    # 
    k_list = [4, 9]
    evaluations = ["CC", "CF", "FF"]
    penalties = ["none", "l2", "l1"]
    frgt_list = [250, 500]

    # coverage
    if exp_type == "cov":
        program = "mlr_eval_sim_coverage.py"
        exp_code = "mlr_coverages"
        exp_mini = "C"
        job_config.set("seq_rep", "fragment_cov", "0.1, 1, 2, 10")

    # mutation
    elif exp_type == "mut":
        program = "mlr_eval_sim_evoparams.py"
        exp_code = "mlr_mutation"
        exp_mini = "M"
        job_config.set("simulation", "mutation_rate", "0.0001, 0.001, 0.01, 0.1")

    # indel
    elif exp_type == "ind":
        program = "mlr_eval_sim_evoparams.py"
        exp_code = "mlr_indel"
        exp_mini = "I"
        job_config.set("simulation", "indel_prob", "0.0001, 0.001, 0.01, 0.1")

    # recombination
    elif exp_type == "rec":
        program = "mlr_eval_sim_evoparams.py"
        exp_code = "mlr_recomb"
        exp_mini = "R"
        job_config.set("simulation", "rep_dual_infection", "0.01")
        job_config.set("simulation", "rep_recombination", "0.0001, 0.001, 0.01, 0.1")

    # imbalanced dataset
    elif exp_type == "dat":
        program = "mlr_eval_sim_imbalanced_dataset.py"
        exp_code = "mlr_imbdata"
        exp_mini = "D"
        job_config.set("simulation", "class_pop_size_std", "0, 5, 10, 50")

    # imbalanced sampling
    elif exp_type == "samp":
        program = "mlr_eval_sim_imbalanced_sampling.py"
        exp_code = "mlr_imbsamp"
        exp_mini = "S"
        job_config.set("seq_rep", "class_size_std", "0, 5, 10, 50")

    # k lengths 
    elif exp_type == "klen":        
        k_list = ["4, 5, 6, 7, 8, 9, 10"]
        #k_list = [4, 5, 6, 7, 8, 9, 10]
        program = "mlr_eval_sim_klen.py"
        exp_code = "mlr_klens"
        exp_mini = "K"
        #job_config.set("seq_rep", "k", k_list[0])

    # Lambda
    elif exp_type == "lambda":
        program = "mlr_eval_sim_lambda.py"
        exp_code = "mlr_lambdas"
        exp_mini = "A"
        job_config.set("classifier", "lambda", "1e-3, 1e-1, 1, 1e1, 1e2")

    # Learning rate
    elif exp_type == "lr":
        program = "mlr_eval_sim_learning_rate.py"
        exp_code = "mlr_learning_rates"
        exp_mini = "L"
        job_config.set("classifier", "learning_rate", "1e-5, 1e-3, 1e-1")

    # Low variance threshold
    elif exp_type == "lowv":
        program = "mlr_eval_sim_lowVar.py"
        exp_code = "mlr_lowvars"
        exp_mini = "V"
        job_config.set("seq_rep", "low_var_threshold", "0.01, 0.1, 0.3, 0.5")

    # number of classes
    elif exp_type == "nbc":
        program = "mlr_eval_sim_nbclasses.py"
        exp_code = "mlr_nbclasses"
        exp_mini = "N"
        job_config.set("simulation", "nb_classes", "2, 5, 10, 20")


    # Output folders
    # ###############
    #job_dir = "jobs_folder/{}/".format(exp_code)
    job_dir = os.path.join(output_folder, "jobs_folder", exp_code)
    makedirs(job_dir, mode=0o700, exist_ok=True)
    s_error = os.path.join(job_dir, "%j.err")
    s_output = os.path.join(job_dir, "%j.out")

    #config_dir = "config_folder/{}/".format(exp_code)
    config_dir = os.path.join(output_folder, "config_folder", exp_code)
    makedirs(config_dir, mode=0o700, exist_ok=True)

    job_config.set("job", "job_code", exp_code)
    job_config.set("io", "outdir", output_folder)

    for eval_type in evaluations:
        if eval_type in ["CF", "FF"]:
            fgt_sizes = frgt_list
        else:
            fgt_sizes = [0] # not important

        for frgt_size, k, pen in product(fgt_sizes, k_list, penalties):
            exp_name = "{}_{}_{}_{}_{}".format(k, eval_type, frgt_size, pen)

            # Update config parser
            job_config.set('evaluation', 'eval_type', eval_type) 
            job_config.set('seq_rep', 'k', k) 
            job_config.set('seq_rep', 'fragment_size', frgt_size) 
            job_config.set('classifier', 'penalty', penalties)

            # write it on a file
            config_file = os.path.join(config_dir, "{}_{}.ini"\
                    "".format(exp_name, str_time))

            with open (config_file, "wb") as fh:
                job_config.write(config_file)

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
            print(cmd)
            #os.system(cmd)


if __name__ == '__main__':
    main(sys.argv)

