#!/usr/bin/env python
import os
import re
from lxml import etree as et
from Bio import SeqIO, Phylo
from random import choice
import numpy as np
from scipy.cluster import hierarchy

__author__ = "nicolas"

class SantaSim():
    """docstring for santaSim."""

    def __init__(self, fastaFile, clsFile, configFile, outDir, outName, virusName = "virus", repeat = 1):
        self.fastaFile_ = str(fastaFile)
        self.clsFile_ = str(clsFile)
        self.configFile_ = str(configFile)
        self.outDir_ = str(outDir)
        self.virusName_ = str(virusName)
        self.repeat_ = str(repeat)
        self.populationSize_ = len(list(SeqIO.parse(self.fastaFile_, "fasta")))
        self.santaPath_ = "{}/santa.jar".format(os.path.dirname(os.path.realpath(__file__)))
        self.outName_ = str(outName)

    # Main function for executing santaSim
    def santaSim(self):
        simFile, treeFile = self.writeXMLConfig()
        cmd = "java -jar {} {}".format(self.santaPath_, self.configFile_)
        print("Executing simulations : " + cmd)
        os.system(cmd)
        self.generateClassesFile(treeFile)
        return simFile

    # Extract nodes name list and distance matrix from tree
    def generateDistanceMatrix(self, tree):
        allclades = list(tree.find_clades(order="level"))
        seqnames = []
        j = 0
        for i in range(len(allclades)):
            if not allclades[i].name:
                allclades[i].name = 'Clade{}'.format(str(j))
                j +=1
            seqnames.append(allclades[i].name)
        lookup = {}
        for i, elem in enumerate(allclades):
            lookup[elem] = i
        distmat = np.zeros((len(allclades),len(allclades)))
        for parent in tree.find_clades(order="level"):
            for child in parent.clades:
                if child.branch_length:
                    distmat[lookup[parent], lookup[child]] = child.branch_length
                    distmat[len(allclades) - lookup[child] - 1, len(allclades) - lookup[parent] - 1] = child.branch_length
        return seqnames, distmat

    # Generate classes file from a tree file in newick format
    def generateClassesFile(self, treeFile):
        t = Phylo.read(treeFile,"newick")
        n, m = self.generateDistanceMatrix(t)
        linkage = hierarchy.ward(m)
        cutree = hierarchy.cut_tree(linkage, n_clusters = [5])
        cls = cutree.tolist()
        cls = [y for x in cls for y in x]
        for i in range(len(cls)):
            if cls.count(cls[i]) < 2:
                cls[i] = 1
        for i in range(0,5):
            print("Class {} : {}".format(str(i), str(cls.count(i))))
        with open(self.clsFile_,"w") as fh:
            for i in range(len(n)):
                if "Clade" not in n[i]:
                    print(n[i], cls[i], sep = ",", file = fh)
            fh.close()

    # Change uncertain nucleotides from sequencing or consensus in databases
    def normaliseNucleotide(self, sequence):
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

    # Write XML configuration file for santaSim
    def writeXMLConfig(self):
        parser = et.XMLParser(remove_blank_text=False)
        root = et.Element("santa")
        replicates = et.SubElement(root, "replicates")
        replicates.text = str(self.repeat_)
        simulation = et.SubElement(root, "simulation")

        fasta = et.SubElement(simulation, "genome")
        seq = choice(list(SeqIO.parse(self.fastaFile_, "fasta")))
        fasta_length = et.SubElement(fasta, "length")
        fasta_length.text = str(len(seq.seq))
        fasta_feature = et.SubElement(fasta, "feature")
        fasta_feature_name = et.SubElement(fasta_feature, "name")
        fasta_feature_name.text = str(seq.id)
        fasta_feature_type = et.SubElement(fasta_feature, "type")
        fasta_feature_type.text = "nucleotide"
        fasta_feature_coords = et.SubElement(fasta_feature, "coordinates")
        fasta_feature_coords.text = "1-{}".format(str(len(seq.seq)))
        fasta_sq = et.SubElement(fasta, "sequences")
        fasta_sq.text = str(self.normaliseNucleotide(seq.seq))

        pop = et.SubElement(simulation, "population")
        pop_size = et.SubElement(pop, "populationSize")
        pop_size.text = str(int(self.populationSize_) * 10)
        pop_inoculum = et.SubElement(pop, "inoculum")
        pop_inoculum.text = "random"

        fitness = et.SubElement(simulation, "fitnessFunction")
        fitness_freq = et.SubElement(fitness, "frequencyDependentFitness")
        fitness_freq_feature = et.SubElement(fitness_freq, "feature")
        fitness_freq_feature.text = "genome"
        fitness_freq_shape = et.SubElement(fitness_freq, "shape")
        fitness_freq_shape.text = "0.5"

        replication = et.SubElement(simulation, "replicator")
        replication_type = et.SubElement(replication, "recombinantReplicator")
        replication_type_dualInfection = et.SubElement(replication_type, "dualInfectionProbability")
        replication_type_dualInfection.text = "0.05"
        replication_type_recombination = et.SubElement(replication_type, "recombinationProbability")
        replication_type_recombination.text = "0.01"

        mutation = et.SubElement(simulation, "mutator")
        mutation_type = et.SubElement(mutation, "nucleotideMutator")
        mutation_type_rate = et.SubElement(mutation_type, "mutationRate")
        mutation_type_rate.text = "0.1"
        mutation_type_bias = et.SubElement(mutation_type, "transitionBias")
        mutation_type_bias.text = "5.0"

        first_epoch = et.SubElement(simulation, "epoch")
        epoch_name = et.SubElement(first_epoch, "name")
        epoch_name.text = "Simulation epoch 1"
        epoch_gen = et.SubElement(first_epoch, "generationCount")
        epoch_gen.text = "100"

        sampling = et.SubElement(simulation, "samplingSchedule")
        sampling_sampler = et.SubElement(sampling, "sampler")
        sampling_sampler_generation = et.SubElement(sampling_sampler, "atGeneration")
        sampling_sampler_generation.text = str(int(self.populationSize_/10))
        sampling_sampler_file = et.SubElement(sampling_sampler, "fileName")
        sampling_sampler_file.text = "{}/{}.fa".format(self.outDir_, self.outName_)
        sampling_sampler_alignment = et.SubElement(sampling_sampler, "alignment")
        sampling_sampler_alignment_size = et.SubElement(sampling_sampler_alignment, "sampleSize")
        sampling_sampler_alignment_size.text = str(self.populationSize_)
        sampling_sampler_alignment_format = et.SubElement(sampling_sampler_alignment, "format")
        sampling_sampler_alignment_format.text = "FASTA"
        sampling_sampler_alignment_label = et.SubElement(sampling_sampler_alignment, "label")
        sampling_sampler_alignment_label.text = "seq_%g_%s"
        sampling_sampler = et.SubElement(sampling, "sampler")
        sampling_sampler_generation = et.SubElement(sampling_sampler, "atGeneration")
        sampling_sampler_generation.text = str(int(self.populationSize_/10))
        sampling_sampler_tree = et.SubElement(sampling_sampler, "fileName")
        sampling_sampler_tree.text = "{}/{}.nh".format(self.outDir_, self.outName_)
        sampling_sampler_alignment = et.SubElement(sampling_sampler, "tree")
        sampling_sampler_alignment_size = et.SubElement(sampling_sampler_alignment, "sampleSize")
        sampling_sampler_alignment_size.text = str(self.populationSize_)
        sampling_sampler_alignment_format = et.SubElement(sampling_sampler_alignment, "format")
        sampling_sampler_alignment_format.text = "NEWICK"
        sampling_sampler_alignment_label = et.SubElement(sampling_sampler_alignment, "label")
        sampling_sampler_alignment_label.text = "seq_%g_%s"
        et.ElementTree(root).write(self.configFile_, pretty_print=True, encoding='utf-8', xml_declaration=False)

        return str(sampling_sampler_file.text), str(sampling_sampler_tree.text)

#Testing
for i in range(20):
    test = SantaSim("/home/nicolas/github/mlr_kgenomvir/data/viruses/HBV01/data.fa", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation/test.csv", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation/test.xml", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation", "sim_test", virusName = "HBV01", repeat = 1)
    test.santaSim()
