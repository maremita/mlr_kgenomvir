#!/usr/bin/env python

from mlr_kgenomvir.data.seq_collections import SeqCollection

import os
import re
import copy
from random import choice
from collections import defaultdict

from lxml import etree as et
from Bio import SeqIO, Phylo
import numpy as np
from scipy.cluster import hierarchy

from joblib import Parallel, delayed

__author__ = "nicolas"


class SantaSim():
    """docstring for santaSim."""

    santaPath = "{}/santa.jar".format(os.path.dirname(os.path.realpath(__file__)))
    print(santaPath)

    def __init__(self, initSeqs, nbClasses, evoParams, outDir, outName, verbose=0):
        self.initSeqs_ = initSeqs
        self.nbClasses_ = nbClasses
        self.evoParams_ = copy.deepcopy(evoParams)
        self.outDir_ = outDir
        self.outName_ = outName
        self.finalFasta_ = os.path.join(self.outDir_, self.outName_+".fa")
        self.finalClsFile_ = os.path.join(self.outDir_, self.outName_+".csv")
        self.verbose_ = verbose
        
        if not isinstance(self.initSeqs_, list):
            raise TypeError("initSeqs_ should be a list of seqRecords")

        if len(self.initSeqs_) > 1:
            # the number of initial sequences should match the number 
            # of desired classes to generate unless if we want to 
            # simulate the whole data from one initial sequence
            assert(len(self.initSeqs_) == self.nbClasses_)

    def __call__(self): 
        # if one sequence is given, we simulate and pick
        # nbClasses sequences to be ancestral sequences
        if len(self.initSeqs_) == 1:
            self.evoParams_["generationCount"] = self.evoParams_["generationCount"]//2
            initOutput = os.path.join(self.outDir_, self.outName_+"_init")
            initFasta, initTree = self.santaSim(self.initSeqs_[0], "cinit", initOutput, self.evoParams_)

            init_seq_col = SeqCollection.read_bio_file(initFasta)
            init_seq_cls = self.generateClasses(initTree, self.nbClasses_)
            print(init_seq_cls.keys())
            seq_names = []
            ancestral_seqs = []
            for c in init_seq_cls:
                seq_names.append(choice(init_seq_cls[c]))

            for seqRec in init_seq_col:
                if seqRec.id in seq_names:
                    ancestral_seqs.append(seqRec)

        # if several sequences are given, each sequence is considered as
        # ancestral sequence for it class
        else:
            ancestral_seqs = self.initSeqs_
        
        # Run Santasim for each ancestral sequence
        parallel = Parallel(n_jobs=self.nbClasses_, prefer="processes", verbose=self.verbose_)
        output = os.path.join(self.outDir_, self.outName_+"_")

        simFiles = parallel(delayed(self.santaSim)(seq, 
            "c{}".format(i), output+str(i), self.evoParams_)
                for i, seq in enumerate(ancestral_seqs))

        # each fasta file corresponds to a class
        # merge dataset
        labeled_seqs = []
        for c, (simFasta, _) in enumerate(simFiles):
            seqData = SeqCollection.read_bio_file(simFasta)
            for seqRec in seqData: seqRec.label = "c{}".format(c)
            labeled_seqs.extend(seqData)

        sim_col = SeqCollection(labeled_seqs)
        sim_col.write(self.finalFasta_, self.finalClsFile_)

        return self.finalFasta_, self.finalClsFile_

    # Main function for executing santaSim
    @classmethod
    def santaSim(cls, sequence, tag, output, evo_params):
        simFasta, simTree, configFile = cls.writeSimXMLConfig(sequence, tag, output, **evo_params)
        cmd = "java -jar {} {}".format(cls.santaPath, configFile)
        print("Executing simulations : " + cmd)
        os.system(cmd)
        return simFasta, simTree

    # Extract nodes name list and distance matrix from tree
    @classmethod
    def generateDistanceMatrix(cls, tree):
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
    @classmethod
    def generateClasses(cls, treeFile, nbClusters):

        class_list = defaultdict(list)
        t = Phylo.read(treeFile,"newick")
        names, matrix = cls.generateDistanceMatrix(t)
        linkage = hierarchy.ward(matrix)
        cutree = hierarchy.cut_tree(linkage, n_clusters = [nbClusters+1])
        # +1 because the first cluster will contain only the root clade
        #print(cutree)
        classes = np.squeeze(cutree.tolist(), axis=1)
        #print(classes)

        for i, c in enumerate(classes):
            if "Clade" not in names[i]:
                class_list[c].append(names[i])
            else:
                print(c, names[i])

        return class_list

    # Change uncertain nucleotides from sequencing or consensus in databases
    @classmethod
    def normaliseNucleotide(cls, sequence):
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
    @classmethod
    def writeSimXMLConfig(
            cls, 
            sequence,
            tag,
            outputFile, # out_dir + out_name [+suffix]
            populationSize=100,
            generationCount=100,
            fitnessFreq=0.5,
            repDualInfection=0.05,
            repRecombination=0.01,
            mutationRate=0.1,
            transitionBias=5.0):

        configFile = "{}.xml".format(outputFile)

        parser = et.XMLParser(remove_blank_text=False)
        root = et.Element("santa")
        replicates = et.SubElement(root, "replicates")
        replicates.text = "1"
        simulation = et.SubElement(root, "simulation")

        fasta = et.SubElement(simulation, "genome")
        fasta_length = et.SubElement(fasta, "length")
        fasta_length.text = str(len(sequence.seq))
        fasta_feature = et.SubElement(fasta, "feature")
        fasta_feature_name = et.SubElement(fasta_feature, "name")
        fasta_feature_name.text = str(sequence.id)
        fasta_feature_type = et.SubElement(fasta_feature, "type")
        fasta_feature_type.text = "nucleotide"
        fasta_feature_coords = et.SubElement(fasta_feature, "coordinates")
        fasta_feature_coords.text = "1-{}".format(str(len(sequence.seq)))
        fasta_sq = et.SubElement(fasta, "sequences")
        fasta_sq.text = str(cls.normaliseNucleotide(sequence.seq))

        pop = et.SubElement(simulation, "population")
        pop_size = et.SubElement(pop, "populationSize")
        pop_size.text = str(populationSize)
        pop_inoculum = et.SubElement(pop, "inoculum")
        pop_inoculum.text = "random"

        fitness = et.SubElement(simulation, "fitnessFunction")
        fitness_freq = et.SubElement(fitness, "frequencyDependentFitness")
        fitness_freq_feature = et.SubElement(fitness_freq, "feature")
        fitness_freq_feature.text = "genome"
        fitness_freq_shape = et.SubElement(fitness_freq, "shape")
        fitness_freq_shape.text = str(fitnessFreq)

        replication = et.SubElement(simulation, "replicator")
        replication_type = et.SubElement(replication, "recombinantReplicator")
        replication_type_dualInfection = et.SubElement(replication_type, "dualInfectionProbability")
        replication_type_dualInfection.text = str(repDualInfection)
        replication_type_recombination = et.SubElement(replication_type, "recombinationProbability")
        replication_type_recombination.text = str(repRecombination)

        mutation = et.SubElement(simulation, "mutator")
        mutation_type = et.SubElement(mutation, "nucleotideMutator")
        mutation_type_rate = et.SubElement(mutation_type, "mutationRate")
        mutation_type_rate.text = str(mutationRate)
        mutation_type_bias = et.SubElement(mutation_type, "transitionBias")
        mutation_type_bias.text = str(transitionBias)

        first_epoch = et.SubElement(simulation, "epoch")
        epoch_name = et.SubElement(first_epoch, "name")
        epoch_name.text = "Simulation epoch 1"
        epoch_gen = et.SubElement(first_epoch, "generationCount")
        epoch_gen.text = str(generationCount)

        sampling = et.SubElement(simulation, "samplingSchedule")
        sampling_sampler = et.SubElement(sampling, "sampler")
        sampling_sampler_generation = et.SubElement(sampling_sampler, "atGeneration")
        sampling_sampler_generation.text = str(generationCount)
        sampling_sampler_file = et.SubElement(sampling_sampler, "fileName")
        sampling_sampler_file.text = "{}.fa".format(outputFile)
        sampling_sampler_alignment = et.SubElement(sampling_sampler, "alignment")
        sampling_sampler_alignment_size = et.SubElement(sampling_sampler_alignment, "sampleSize")
        sampling_sampler_alignment_size.text = str(populationSize)
        sampling_sampler_alignment_format = et.SubElement(sampling_sampler_alignment, "format")
        sampling_sampler_alignment_format.text = "FASTA"
        sampling_sampler_alignment_label = et.SubElement(sampling_sampler_alignment, "label")
        sampling_sampler_alignment_label.text = "seq_%g_%s"+"_{}".format(tag)

        sampling_sampler = et.SubElement(sampling, "sampler")
        sampling_sampler_generation = et.SubElement(sampling_sampler, "atGeneration")
        sampling_sampler_generation.text = str(generationCount)
        sampling_sampler_tree = et.SubElement(sampling_sampler, "fileName")
        sampling_sampler_tree.text = "{}.nh".format(outputFile)
        sampling_sampler_alignment = et.SubElement(sampling_sampler, "tree")
        sampling_sampler_alignment_size = et.SubElement(sampling_sampler_alignment, "sampleSize")
        sampling_sampler_alignment_size.text = str(populationSize)
        sampling_sampler_alignment_format = et.SubElement(sampling_sampler_alignment, "format")
        sampling_sampler_alignment_format.text = "NEWICK"
        sampling_sampler_alignment_label = et.SubElement(sampling_sampler_alignment, "label")
        sampling_sampler_alignment_label.text = "seq_%g_%s"+"_{}".format(tag)

        et.ElementTree(root).write(configFile, pretty_print=True, encoding='utf-8', xml_declaration=False)

        return str(sampling_sampler_file.text), str(sampling_sampler_tree.text), configFile

#Testing
#for i in range(20):
#    test = SantaSim("/home/nicolas/github/mlr_kgenomvir/data/viruses/HBV01/data.fa", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation/test.csv", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation/test.xml", "/home/nicolas/github/mlr_kgenomvir/mlr_kgenomvir/simulation", "sim_test", virusName = "HBV01", repeat = 1)
#    print(test.santaSim())
