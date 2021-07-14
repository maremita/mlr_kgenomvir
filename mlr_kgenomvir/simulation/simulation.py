#!/usr/bin/env python
#SANTA-SIM
import os
import re
from lxml import etree as et
from Bio import SeqIO
from random import choice

"""
#pyvolve
import pyvolve
import ngesh
from Bio import SeqIO
"""
__author__ = "nicolas"

"""
def pyvolve_sim(virus_name, n_iter, virus_file, outdir):
    for i in range(n_iter):
        #Random phylogeny tree to simulate
        my_tree = pyvolve.read_tree(tree = ngesh.random_tree.fasta_tree(birth = 1.0, death = 0.5, max_time = 3.0, labels = "enum").write(format = 1))
        #Evolutionary model
        my_model = pyvolve.Model("nucleotide")
        #Partitions
        my_partition = pyvolve.Partition(models = my_model, root_sequence = str(list(SeqIO.parse(virus_file,"fasta"))[0].seq))
        #Evolve partitions
        my_evolver = pyvolve.Evolver(tree = my_tree, partitions = my_partition)
        outfile = str(outdir) + "/{}_simulation_{}".format(str(virus_name),str(i))
        my_evolver(ratefile = None, infofile = None, seqfile = outfile)
"""

#to add to mlr_eval_cv_*

    #Simulation
    #sim_type = config.getstr("simulation", "type")
    #sim_iter = config.getint("simulation", "iterations")len()
    #sim_tree = config.getstr("simulation", "phylogenic_tree")
    #sim_dir = config.getstr("simulation", "directory")

def santa_sim(fastaFile, configFile, outDir, virusName, repeat = 1, populationSize = 1000):
    write_xml_config(fastaFile, configFile, outDir, virusName, repeat, populationSize)
    cmd = "java -jar santa.jar {}".format(configFile)
    print("Executing simulations :" + cmd)
    os.system(cmd)

def normalise_nucleotide(sequence):
    seq = list(sequence)
    for i in range(len(seq)):
        if not bool(re.match('^[ACGT]+$', str(seq[i]))):
            if seq[i] == "R":
                seq[i] = choice(["A","G"])
            elif seq[i] == "Y":
                seq[i] = choice(["C","T"])
            elif seq[i] == "K":
                seq[i] = choice(["G","T"])
            elif seq[i] == "M":
                seq[i] = choice(["A","C"])
            elif seq[i] == "S":
                seq[i] = choice(["C","G"])
            elif seq[i] == "W":
                seq[i] = choice(["A","T"])
            elif seq[i] == "B":
                seq[i] = choice(["C","G","T"])
            elif seq[i] == "D":
                seq[i] = choice(["A","G","T"])
            elif seq[i] == "H":
                seq[i] = choice(["A","C","T"])
            elif seq[i] == "V":
                seq[i] = choice(["A","C","G"])
            elif seq[i] == "N":
                seq[i] = choice(["C","T","A","G"])
            elif seq[i] == "X":
                seq[i] = choice(["C","T","A","G"])
    sequence = "".join(seq)
    return sequence

def write_xml_config(infile, outfile, outDir, virusName = "virus", repeat = 1, populationSize = 100):
    parser = et.XMLParser(remove_blank_text=False)
    root = et.Element("santa")
    replicates = et.SubElement(root, "replicates")
    replicates.text = str(repeat)
    simulation = et.SubElement(root, "simulation")

    fasta = et.SubElement(simulation, "genome")
    fasta_file = list(SeqIO.parse(infile, "fasta"))
    list_len = []
    for i in range(len(fasta_file)):
        list_len.append(len(fasta_file[i].seq))
    for i in range(len(fasta_file)):
        fasta_length = et.SubElement(fasta, "length")
        fasta_length.text = str(min(list_len))
        fasta_feature = et.SubElement(fasta, "feature")
        fasta_feature_name = et.SubElement(fasta_feature, "name")
        fasta_feature_name.text = str(virusName)
        fasta_feature_type = et.SubElement(fasta_feature, "type")
        fasta_feature_type.text = "nucleotide"
        fasta_feature_coords = et.SubElement(fasta_feature, "coordinates")
        fasta_feature_coords.text = "1-" + str(fasta_length.text)
    fasta_sq = et.SubElement(fasta, "sequences")
    fasta_list = [""]
    for i in range(len(fasta_file)):
        fasta_list.append(str(">" + fasta_file[i].id))
        fasta_list.append(str(normalise_nucleotide(fasta_file[i].seq[0:min(list_len)])))
    fasta_sq.text = str("\n".join(fasta_list))

    pop = et.SubElement(simulation, "population")
    pop_size = et.SubElement(pop, "populationSize")
    pop_size.text = str(int(populationSize) * 10)
    pop_inoculum = et.SubElement(pop, "inoculum")
    pop_inoculum.text = "random"

    fitness = et.SubElement(simulation, "fitnessFunction")
    fitness_freq = et.SubElement(fitness, "frequencyDependentFitness")
    fitness_freq_feature = et.SubElement(fitness_freq, "feature")
    fitness_freq_feature.text = "genome"
    fitness_freq_shape = et.SubElement(fitness_freq, "shape")
    fitness_freq_shape.text = "0.5"

    replication = et.SubElement(simulation, "replicator")
    replication_type = et.SubElement(replication, "clonalReplicator")

    mutation = et.SubElement(simulation, "mutator")
    mutation_type = et.SubElement(mutation, "nucleotideMutator")
    mutation_type_rate = et.SubElement(mutation_type, "mutationRate")
    mutation_type_rate.text = "1.0E-4"
    mutation_type_bias = et.SubElement(mutation_type, "transitionBias")
    mutation_type_bias.text = "2.0"

    #Indel migh cause problems for computing (memory + time + other parameters)
    mutation_type_indel = et.SubElement(mutation_type, "indelmodel")
    mutation_type_indel.set("model","NB")
    mutation_type_indel.text = "0.4 1"
    mutation_type_insert = et.SubElement(mutation_type, "insertprob")
    mutation_type_insert.text = "2.5E-2"
    mutation_type_delete = et.SubElement(mutation_type, "deleteprob")
    mutation_type_delete.text = "2.5E-2"

    first_epoch = et.SubElement(simulation, "epoch")
    epoch_name = et.SubElement(first_epoch, "name")
    epoch_name.text = "Simulation epoch 1"
    epoch_gen = et.SubElement(first_epoch, "generationCount")
    epoch_gen.text = "100"

    sampling = et.SubElement(simulation, "samplingSchedule")
    sampling_sampler = et.SubElement(sampling, "sampler")
    sampling_sampler_frequency = et.SubElement(sampling_sampler, "atFrequency")
    sampling_sampler_frequency.text = str(int(populationSize/10))
    sampling_sampler_file = et.SubElement(sampling_sampler, "fileName")
    sampling_sampler_file.text = str(outDir + "/simulation_%r.fa")
    sampling_sampler_alignment = et.SubElement(sampling_sampler, "alignment")
    sampling_sampler_alignment_size = et.SubElement(sampling_sampler_alignment, "sampleSize")
    sampling_sampler_alignment_size.text = str(populationSize)
    sampling_sampler_alignment_format = et.SubElement(sampling_sampler_alignment, "format")
    sampling_sampler_alignment_format.text = "FASTA"
    sampling_sampler_alignment_label = et.SubElement(sampling_sampler_alignment, "label")
    sampling_sampler_alignment_label.text = "seq_%g_%s"
    et.ElementTree(root).write(outfile, pretty_print=True, encoding='utf-8', xml_declaration=False)


#testing
santa_sim("/home/nicolas/github/mlr_kgenomvir/data/viruses/HBV01/data.fa", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation/test.xml", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation")
