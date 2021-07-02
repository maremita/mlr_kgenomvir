#!/usr/bin/env python
import pyvolve
import ngesh
from Bio import SeqIO

__author__ = "nicolas"

def evolutive(virus_name, n_iter, virus_file, outdir):
    for i in range(n_iter):
        #Random phylogeny tree to simulate
        my_tree = pyvolve.read_tree(tree = ngesh.random_tree.gen_tree(birth = 1.0, death = 0.5, max_time = 3.0, labels = "enum").write(format = 1))
        #Evolutionary model
        my_model = pyvolve.Model("nucleotide")
        #Partitions
        my_partition = pyvolve.Partition(models = my_model, root_sequence = str(list(SeqIO.parse(virus_file,"fasta"))[0].seq))
        #Evolve partitions
        my_evolver = pyvolve.Evolver(tree = my_tree, partitions = my_partition)
        outfile = str(outdir) + "/{}_simulation_{}".format(str(virus_name),str(i))
        my_evolver(ratefile = None, infofile = None, seqfile = outfile)

#Test
#evolutive("HBV01", 10, "/home/nicolas/github/mlr_kgenomvir/data/viruses/HBV01/data.fa", "/home/nicolas/github/metagenomics_ML_exp/local/results/test/simulation")

#to add to mlr_eval_cv_*

    #Simulation
    sim_type = config.getstr("simulation", "type")
    sim_iter = config.getint("simulation", "iterations")
    sim_tree = config.getstr("simulation", "phylogenic_tree")
    sim_dir = config.getstr("simulation", "directory")
