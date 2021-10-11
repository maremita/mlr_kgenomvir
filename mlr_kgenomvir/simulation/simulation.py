#!/usr/bin/env python

from mlr_kgenomvir.data.seq_collections import SeqCollection

import os
import re
import copy
from random import choice
from collections import defaultdict

import dendropy
from phylodm.pdm import PDM
from lxml import etree as et

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.stats import norm

from joblib import Parallel, delayed

__author__ = "nicolas, amine"


class SantaSim():
    """docstring for santaSim."""

    santaPath = "{}/santa.jar".format(
            os.path.dirname(os.path.realpath(__file__)))

    def __init__(self, 
            initSeqs,
            initGenCountFrac,
            nbClasses,
            classPopSize,
            evoParams,
            outDir,
            outName,
            classPopSizeStd=None,
            classPopSizeMax=1000,
            classPopSizeMin=5,
            load_data=False,
            random_state=None,
            verbose=0):

        self.initSeqs_ = initSeqs
        self.initGenCountFrac_ = initGenCountFrac
        self.nbClasses_ = nbClasses
        self.classPopSize_ = classPopSize
        self.classPopSizeStd_ = classPopSizeStd
        self.classPopSizeMax_ = classPopSizeMax
        self.classPopSizeMin_ = classPopSizeMin
        self.evoParams_ = copy.deepcopy(evoParams)
        self.outDir_ = outDir
        self.outName_ = outName
        self.loadData_ = load_data
        self.random_state_ = random_state
        self.verbose_ = verbose
 
        self.finalFasta_ = os.path.join(self.outDir_, 
                self.outName_+".fa")
        self.finalClsFile_ = os.path.join(self.outDir_, 
                self.outName_+".csv")


        if not isinstance(self.initSeqs_, list):
            raise TypeError(
                    "initSeqs_ should be a list of seqRecords")

        if len(self.initSeqs_) > 1:
            # the number of initial sequences should match the number 
            # of desired classes to generate unless if we want to 
            # simulate the whole data from one initial sequence
            assert(len(self.initSeqs_) == self.nbClasses_)

        # 
        assert(self.classPopSizeMax_ >= self.classPopSizeMin_)

    def __call__(self):
        if self.loadData_ and os.path.isfile(self.finalFasta_) \
                and os.path.isfile(self.finalClsFile_): 
            if self.verbose_:
                print("\nLoading simulated sequences from files",
                        flush=True)
            # Don't run simulation and load
            # previous simulated data
            return self.finalFasta_, self.finalClsFile_

        # Flag to sample each class using a class-size-based
        # normal distribution
        if isinstance(self.classPopSizeStd_, (int, float)):
            sample_classes = True
        else:
            sample_classes = False

        # If one sequence is given, we simulate and pick
        # nbClasses sequences to be ancestral sequences
        if len(self.initSeqs_) == 1:
            # initialize generation count for initial simulation
            genCount = self.evoParams_["generationCount"]
            self.evoParams_["generationCount"] =\
                    int(genCount * self.initGenCountFrac_)

            initOutput = os.path.join(self.outDir_,
                    self.outName_+"_init")
            initFasta, initTree = self.santaSim(self.initSeqs_[0],
                    "cinit", initOutput, self.evoParams_,
                    set_seed=None)

            init_seq_col = SeqCollection.read_bio_file(initFasta)
            init_seq_cls = self.generateClasses(initTree, 
                    self.nbClasses_)

            seq_names = []
            ancestral_seqs = []
            for c in init_seq_cls:
                seq_names.append(choice(init_seq_cls[c]))

            for seqRec in init_seq_col:
                if seqRec.id in seq_names:
                    ancestral_seqs.append(seqRec)

            # Set generation count for subsequent simulation
            self.evoParams_["generationCount"] =\
                    int(genCount * (1 - self.initGenCountFrac_))

        # If several sequences are given, each sequence
        # is considered as ancestral sequence for it class.
        # Also we use the whole generation count 
        else:
            ancestral_seqs = self.initSeqs_

        if sample_classes:
            # Random Sampling of class sizes to create
            # [im]balanced dataset
            # Choosing class sizes is random and follows a normal
            # distribution
            #
            min_ = self.classPopSizeMin_
            max_ = self.classPopSizeMax_
            lim_fun = lambda e: min_ if e < min_ else\
                    (max_ if (e > max_) else e)
            #
            pop_sizes = list(map(lim_fun, norm.rvs(
                loc=self.classPopSize_, 
                scale=self.classPopSizeStd_, 
                size=self.nbClasses_).astype(np.int)))

        else:
            # All classes have the same size
            pop_sizes = [self.classPopSize_]*self.nbClasses_

        if self.verbose_:
            print("\nSimulating dataset with class sizes:"\
                    "\n{}\n".format(pop_sizes), flush=True)
        #
        parallel = Parallel(n_jobs=self.nbClasses_,
                prefer="processes", verbose=self.verbose_)
        output = os.path.join(self.outDir_, self.outName_+"_")

        # Run Santasim for each ancestral sequence
        # to simulate nbClasses
        simFiles = parallel(delayed(self.santaSim)(seq, 
            "c{}".format(i), output+str(i), self.evoParams_,
            set_pop_size=pop_size, set_seed=None) 
            for i, (seq, pop_size) in enumerate(zip(
                ancestral_seqs, pop_sizes)))

        # Merge datasets
        # Each fasta file corresponds to a class
        labeled_seqs = []
        for c, (simFasta, _) in enumerate(simFiles):
            seqData = SeqCollection.read_bio_file(simFasta)
            for seqRec in seqData: seqRec.label = "c{}".format(c)
            labeled_seqs.extend(seqData)

        sim_col = SeqCollection(labeled_seqs)

        # Write fasta and label files
        sim_col.write(self.finalFasta_, self.finalClsFile_)

        return self.finalFasta_, self.finalClsFile_

    # Main function for executing santaSim
    @classmethod
    def santaSim(cls, sequence, tag, output, evo_params, 
            set_pop_size=None, set_seed=None):

        if set_pop_size:
            evo_params["populationSize"] = set_pop_size
 
        simFasta, simTree, configFile = cls.writeSimXMLConfig(
                sequence, tag, output, **evo_params)
        seed_str = ""
        if set_seed:
            seed_str = " -seed={}".format(set_seed)
        cmd = "java -jar {}{} {}".format(cls.santaPath, seed_str,
                configFile)
        print("Executing simulations : " + cmd)
        os.system(cmd)
        return simFasta, simTree

    # Generate classes file from a tree file in newick format
    @classmethod
    def generateClasses(cls, treeFile, nbClusters):
        class_list = defaultdict(list)

        with open(treeFile, 'r') as fh:
            treeStr = fh.read().replace('\n', '')
        # SantSim generate newick without the semicomma at the end
        if not treeStr.endswith(";"): treeStr += ";"

        tree = dendropy.Tree.get(data=treeStr, schema='newick')
        pdm = PDM.get_from_dendropy(tree=tree, method='pd', cpus=1)
        # Attention: dendropy replaces "_" by space in names!
        names, matrix = pdm.as_matrix()
        linkage = hierarchy.ward(pdist(matrix))
        cutree = hierarchy.cut_tree(linkage, 
                n_clusters = [nbClusters])
        classes = np.squeeze(cutree, axis=1)

        for i, c in enumerate(classes):
            class_list[c].append(names[i].replace(" ", "_"))

        return class_list

    # Change uncertain nucleotides from sequencing or 
    # consensus in databases
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
            repRecombination=0.0001,
            mutationRate=0.0001,
            transitionBias=2.0,
            indelModelNB=None,
            indelProb=None):

        configFile = "{}.xml".format(outputFile)

        if isinstance(sequence, (int)):
            # if no sequence is fed and instead we have 
            # the length of the default sequence
            inoculum = "none"
        else:
            inoculum = "all"

        parser = et.XMLParser(remove_blank_text=False)
        root = et.Element("santa")
        replicates = et.SubElement(root, "replicates")
        replicates.text = "1"
        simulation = et.SubElement(root, "simulation")

        fasta = et.SubElement(simulation, "genome")
        fasta_length = et.SubElement(fasta, "length")

        if inoculum == "none":
            # inoculum = none (poly-adenine sequence) is not 
            # implemented in SantaSimi v1.0 yet
            #
            # Instead we generate a random sequence
            ## code from stackoverflow
            bases = ("A", "G", "C", "T")
            probs = (0.2, 0.3, 0.3, 0.2)

            seqrand = ''.join(np.random.choice(bases, p=probs) 
                    for _ in range(sequence))

            fasta_length.text = str(sequence)
            fasta_sq = et.SubElement(fasta, "sequences")
            fasta_sq.text = str(seqrand)
            inoculum = "all"
        else:
            fasta_length.text = str(len(sequence.seq))
            fasta_sq = et.SubElement(fasta, "sequences")
            fasta_sq.text = str(
                    cls.normaliseNucleotide(sequence.seq))

        #
        pop = et.SubElement(simulation, "population")
        pop_size = et.SubElement(pop, "populationSize")
        pop_size.text = str(populationSize)
        pop_inoculum = et.SubElement(pop, "inoculum")
        pop_inoculum.text = inoculum

        #
        fitness = et.SubElement(simulation, "fitnessFunction")
        fitness_freq = et.SubElement(
                fitness, "frequencyDependentFitness")

        fitness_freq_feature = et.SubElement(fitness_freq, "feature")
        fitness_freq_feature.text = "genome"

        fitness_freq_shape = et.SubElement(fitness_freq, "shape")
        fitness_freq_shape.text = str(fitnessFreq)

        #
        replication = et.SubElement(simulation, "replicator")
        replication_type = et.SubElement(
                replication, "recombinantReplicator")

        replication_type_dualInfection = et.SubElement(
                replication_type, "dualInfectionProbability")
        replication_type_dualInfection.text = str(repDualInfection)

        replication_type_recombination = et.SubElement(
                replication_type, "recombinationProbability")
        replication_type_recombination.text = str(repRecombination)

        #
        mutation = et.SubElement(simulation, "mutator")
        mutation_type = et.SubElement(mutation, "nucleotideMutator")

        mutation_type_rate = et.SubElement(
                mutation_type, "mutationRate")
        mutation_type_rate.text = str(mutationRate)

        mutation_type_bias = et.SubElement(
                mutation_type, "transitionBias")
        mutation_type_bias.text = str(transitionBias)

        if indelModelNB is not None and indelProb is not None:
            indel_model_NB = et.SubElement(
                    mutation_type, "indelmodel")
            indel_model_NB.set("model", "NB")
            indel_model_NB.text = str(indelModelNB)

            insert_prob = et.SubElement(
                    mutation_type, "insertprob")
            insert_prob.text = str(indelProb)
 
            delete_prob = et.SubElement(
                    mutation_type, "deleteprob")
            delete_prob.text = str(indelProb)

        #
        first_epoch = et.SubElement(simulation, "epoch")
        epoch_name = et.SubElement(first_epoch, "name")
        epoch_name.text = "Simulation epoch 1"
        epoch_gen = et.SubElement(first_epoch, "generationCount")
        epoch_gen.text = str(generationCount)

        #
        sampling = et.SubElement(simulation, "samplingSchedule")
        sampling_sampler = et.SubElement(sampling, "sampler")
        sampling_sampler_generation = et.SubElement(
                sampling_sampler, "atGeneration")
        sampling_sampler_generation.text = str(generationCount)
        sampling_sampler_file = et.SubElement(
                sampling_sampler, "fileName")
        sampling_sampler_file.text = "{}.fa".format(outputFile)
        sampling_sampler_alignment = et.SubElement(
                sampling_sampler, "alignment")
        sampling_sampler_alignment_size = et.SubElement(
                sampling_sampler_alignment, "sampleSize")
        sampling_sampler_alignment_size.text = str(populationSize)
        sampling_sampler_alignment_format = et.SubElement(
                sampling_sampler_alignment, "format")
        sampling_sampler_alignment_format.text = "FASTA"
        sampling_sampler_alignment_label = et.SubElement(
                sampling_sampler_alignment, "label")
        sampling_sampler_alignment_label.text =\
                "seq_%g_%s"+"_{}".format(tag)

        sampling_sampler = et.SubElement(sampling, "sampler")
        sampling_sampler_generation = et.SubElement(
                sampling_sampler, "atGeneration")
        sampling_sampler_generation.text = str(generationCount)
        sampling_sampler_tree = et.SubElement(
                sampling_sampler, "fileName")
        sampling_sampler_tree.text = "{}.nh".format(outputFile)
        sampling_sampler_alignment = et.SubElement(
                sampling_sampler, "tree")
        sampling_sampler_alignment_size = et.SubElement(
                sampling_sampler_alignment, "sampleSize")
        sampling_sampler_alignment_size.text = str(populationSize)
        sampling_sampler_alignment_format = et.SubElement(
                sampling_sampler_alignment, "format")
        sampling_sampler_alignment_format.text = "NEWICK"
        sampling_sampler_alignment_label = et.SubElement(
                sampling_sampler_alignment, "label")
        sampling_sampler_alignment_label.text =\
                "seq_%g_%s"+"_{}".format(tag)

        #
        et.ElementTree(root).write(configFile, pretty_print=True,
                encoding='utf-8', xml_declaration=False)

        return str(sampling_sampler_file.text),\
                str(sampling_sampler_tree.text), configFile

